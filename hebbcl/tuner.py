import os
from pathlib import Path
import torch
import numpy as np
import random
import argparse
import utils
import hebbcl
import pickle
import ray
import pandas as pd
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest import SearchAlgorithm
from ray.tune.suggest.suggestion import Searcher
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler, TrialScheduler
from typing import Dict, Union, Mapping
from joblib import Parallel, delayed


class HPOTuner(object):
    def __init__(
        self,
        args: argparse.Namespace,
        time_budget: int = 100,
        metric: str = "loss",
        dataset: str = "blobs",
        filepath: str = "/../datasets/",
        filesuffix: str = "_ds18",
        working_dir: str = "../ray_temp_env/",
        log_dir: str = "../ray_logs/",
    ):
        """hyperparameter optimisation for nnets

        Args:
            args (Namespace): collection of neural network training parameters
            time_budget (int, optional): time budget allocated to the fitting process (in seconds). Defaults to 100.
            metric (str, optional): metric to optimise, can be "acc" or "loss". Defaults to "loss".
            dataset (str, optional): which dataset to use. can be trees or blobs. Defaults to "blobs".
            filepath (str, optional): relative path to datasets. Defaults to "../datasets/".
            working_dir (str, optional): relative path to working dir for ray environment.
             Defaults to "../ray_temp_env/"
        """

        self.metric = self._set_metric(metric)
        self.mode = self._set_mode(metric)

        self.time_budget = time_budget
        self.args = args

        self.best_cfg = None
        self.results = None

        self.dataset = dataset
        self.filepath = filepath
        self.filesuffix = filesuffix

        if self.args.hpo_fixedseed:
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

        ray.shutdown()
        ray.init(
            runtime_env={
                "working_dir": working_dir,
                "env_vars": {
                    "TUNE_ORIG_WORKING_DIR": os.getcwd(),
                    "TUNE_RESULT_DIR ": os.getcwd() + log_dir,
                },
                "py_modules": [utils, hebbcl],
            }
        )

    def tune(
        self,
        time_budget: int = None,
        n_samples: int = None,
        resume: bool = False,
        resources_per_trial: Mapping[str, float] = {"cpu": 3, "gpu": 0},
    ):
        """runs the tuner. time budget and n trials set in constructor can be overwritten here

        Args:
            time_budget (int, optional): time in seconds allocated to the fitting proces. Defaults to 100.
            n_samples (int, optional): number of trials. Defaults to 100.
            resume (bool, optional): warm start. Defaults to False.
        """
        # run ray tune
        self._analysis = tune.run(
            lambda cfg, checkpoint_dir: self._trainable(cfg, checkpoint_dir),
            config=self._get_config(),
            time_budget_s=time_budget if time_budget else self.time_budget,
            num_samples=n_samples if n_samples else 100,
            search_alg=self._set_algo(),
            scheduler=self._set_scheduler(),
            metric=self.metric,
            mode=self.mode,
            resources_per_trial=resources_per_trial,
            verbose=1,
            resume=resume,
        )
        self.best_cfg = self._analysis.get_best_config(
            metric=self.metric, mode=self.mode
        )

        # results as dataframe
        self.results = self._analysis.results_df

    def _trainable(self, config: dict, checkpoint_dir: str = None):  # noqa E231
        """training loop for nnet

        Args:
            config (dict): search space config for nnet hyperparams
        """
        self.args.device, _ = utils.nnet.get_device(self.args.cuda)
        # get deterministic behaviour if desired:
        if self.args.hpo_fixedseed:
            np.random.seed(config["seed"])
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
        # replace self.args with ray tune config
        for k, v in config.items():
            setattr(self.args, k, v)

        # get dataset
        if self.dataset == "blobs":
            data = utils.data.make_blobs_dataset(self.args)
        elif self.dataset == "trees":
            datapath = os.environ.get("TUNE_ORIG_WORKING_DIR") + self.filepath

            data = utils.data.make_trees_dataset(
                self.args, filepath=datapath, filesuffix=self.filesuffix
            )

        # instantiate model and hebbcl.trainer.Optimiser
        model = hebbcl.model.ModelFactory.create(self.args)
        optim = hebbcl.trainer.Optimiser(self.args)

        # send model and data to device
        model = model.to(self.args.device)
        x_train = torch.from_numpy(data["x_train"]).float().to(self.args.device)
        y_train = torch.from_numpy(data["y_train"]).float().to(self.args.device)

        x_both = (
            torch.from_numpy(
                np.concatenate((data["x_test_a"], data["x_test_b"]), axis=0)
            )
            .float()
            .to(self.args.device)
        )
        r_both = (
            torch.from_numpy(
                np.concatenate((data["y_test_a"], data["y_test_b"]), axis=0)
            )
            .float()
            .to(self.args.device)
        )

        # loop over data and apply hebbcl.trainer.Optimiser
        idces = np.arange(len(x_train))
        for ii, x, y in zip(idces, x_train, y_train):
            optim.step(model, x, y)

            if ii % self.args.log_interval == 0:
                # obtain loss/acc metrics
                loss_both = utils.nnet.from_gpu(
                    optim.loss_funct(r_both, model(x_both))
                ).ravel()[0]
                acc_both = utils.eval.compute_accuracy(r_both, model(x_both))

                # send metrics to ray tune
                tune.report(mean_loss=loss_both, mean_acc=acc_both)

    def _set_algo(self) -> Union[Searcher, SearchAlgorithm, None]:
        """sets search algorithm based on user preference"""
        if self.args.hpo_searcher == "bohb":
            algo = TuneBOHB(
                metric=self.metric,
                mode=self.mode,
                seed=self.args.seed if self.args.hpo_fixedseed else None,
            )
        else:
            algo = None
        return algo

    def _set_scheduler(self) -> Union[TrialScheduler, None]:
        """sets scheduler based on user preference

        Returns:
            Union[Searcher, None]: search scheduler or none
        """
        if self.args.hpo_scheduler == "bohb":
            scheduler = HyperBandForBOHB(
                time_attr="training_iteration",
                max_t=self.time_budget // 2,
            )
        elif self.args.hpo_scheduler == "asha":
            scheduler = ASHAScheduler(
                time_attr="training_iteration",  # TODO what's this?
                max_t=self.time_budget // 2,
                grace_period=10,
                reduction_factor=3,
                brackets=1,
            )
        else:
            scheduler = None
        return scheduler

    def _get_config(self) -> dict:
        """retrieves model-specific HPO config"""
        if self.args.n_layers == 1:
            if self.args.gating == "SLA":
                config = {
                    "lrate_sgd": tune.loguniform(1e-4, 1e-1),
                    "lrate_hebb": tune.loguniform(1e-4, 1e-1),
                    "normaliser": tune.uniform(1, 20),
                    "n_episodes": tune.choice([200, 500, 1000]),
                    "sla": tune.grid_search([0, 1]),
                }
            elif self.args.gating == "manual":
                config = {
                    "lrate_sgd": tune.loguniform(1e-3, 1e-1),
                    "n_episodes": tune.choice([200, 500, 1000]),
                    "weight_init": tune.loguniform(1e-4, 1e-3),
                }
            elif (self.args.gating == "oja") or (self.args.gating == "oja_ctx"):
                config = {
                    "lrate_sgd": tune.loguniform(1e-4, 1e-1),
                    "lrate_hebb": tune.loguniform(1e-4, 1e-1),
                    "ctx_scaling": tune.randint(1, 8),
                }
            elif self.args.gating is None:
                config = {
                    "lrate_sgd": tune.loguniform(1e-3, 1e-1),
                    "ctx_scaling": tune.randint(1, 8),
                }
        elif self.args.n_layers == 2:
            if self.args.gating is None:
                config = {
                    "lrate_sgd": tune.loguniform(1e-5, 1e-1),
                    "ctx_scaling": tune.randint(1, 8),
                }
            elif self.args.gating == "oja_ctx":
                config = {
                    "lrate_sgd": tune.loguniform(1e-5, 1e-1),
                    "lrate_hebb": tune.loguniform(1e-4, 1e-1),
                    "ctx_scaling": tune.randint(1, 8),
                }
            elif self.args.gating == "oja":
                config = {
                    "lrate_sgd": tune.loguniform(1e-5, 1e-1),
                    "lrate_hebb": tune.loguniform(1e-5, 1e-1),
                    "ctx_scaling": tune.randint(2, 50),
                }
            else:
                raise NotImplementedError(
                    "gating strategy not implemented for two layer net"
                )

        if self.args.hpo_fixedseed:
            config["seed"] = tune.randint(0, 10000)

        return config

    def _set_metric(self, metric: str) -> str:
        """verifies and sets metric chosen by user

        Args:
            metric (str): can be loss or acc

        Raises:
            ValueError: if not loss or acc

        Returns:
            str: chosen metric
        """
        if metric == "loss":
            return "mean_loss"
        elif metric == "acc":
            return "mean_acc"
        else:
            raise ValueError("Invalid metric provided (choose 'loss' or 'acc'")

    def _set_mode(self, metric: str) -> str:
        """sets sign of metric to optimise (search for min or max)

        Args:
            metric (str): which metric. either "loss" or "acc"

        Raises:
            ValueError: if metric neither "loss" nor "acc"

        Returns:
            str: optimisation mode (min or max)
        """
        if metric == "loss":
            return "min"
        elif metric == "acc":
            return "max"
        else:
            raise ValueError("Invalid metric provided (choose 'loss' or 'acc'")


def save_tuner_results(
    df: pd.DataFrame,
    args: argparse.Namespace,
    filename: str = "configs",
    filepath: str = "results/",
) -> argparse.Namespace:
    """saves results from HPOTuner call

    Args:
        df (pd.DataFrame): table with results for individual trials
        args (argparse.Namespace): various configuration parameters
        filename (str, optional): name of file on disk. Defaults to "configs".
        filepath (str, optional): path to file on disk. Defaults to "results".
    """
    # preprocessing ....
    # print(df.columns)
    cols = [
        "mean_loss",
        "mean_acc",
        "done",
    ]
    df = df[[c for c in df if c in cols or c.startswith("config")]]
    # df = df[df["done"] is True]
    # df = df.drop(columns=["done"])
    # df = df.dropna()
    df = df.sort_values("mean_acc", ascending=False)

    args = dict(sorted(vars(args).items(), key=lambda k: k[0]))
    results = {
        "df": df,
        "config": args,
    }
    # and store away ....
    with open(filepath + "raytune_" + filename + ".pkl", "wb") as f:
        pickle.dump(results, f)


def load_tuner_results(
    filename: str, filepath: str = "results/"
) -> Dict[pd.DataFrame, argparse.Namespace]:
    """loads results from hpo tuner run

    Args:
        filename (str): filename of tuner results
        filepath (str, optional): folder containing tuner results. Defaults to "results/".

    Returns:
        Dict[pd.DataFrame, argparse.Namespace]: table with results of individual runs as well as config file
    """

    with open(filepath + "raytune_" + filename + ".pkl", "rb") as f:
        results = pickle.load(f)
        return results


def validate_tuner_results(
    filename: str,
    filepath: str = "./results/",
    datapath: str = "./datasets/",
    datasuffix: str = "_ds18",
    whichtrial: int = 0,
    njobs: int = -1,
):
    """validates results from HPO by running a series of independent training runs
     with randomly initialised weights.
    Stores results to disk

    Args:
        filename (str): name of tuning run
        filepath (str, optional): path to tuning results. Defaults to "./results/".
        datapath (str, optional): path to trees datasets. Defaults to "./datasets/".
        whichtrial (int, optional): trial from HPO df to use. Defaults to 0 (first).

    """

    # load tuner results
    results = load_tuner_results(filename, filepath)
    args = argparse.Namespace(**results["config"])

    # extract best config and set args
    df = results["df"].sort_values("mean_loss")
    params = ["config.lrate_sgd", "config.lrate_hebb", "config.ctx_scaling"]
    hps = dict(df[[c for c in df.columns if c in params]].iloc[whichtrial, :])
    for k, v in hps.items():
        setattr(args, k.split(".")[1], v)

    args.save_dir = filename if whichtrial == 0 else filename + "_" + str(whichtrial)
    dataset = "blobs" if "blobs" in filename else "trees"

    # run jobs in parallel
    seeds = np.random.randint(np.iinfo(np.int32).max, size=args.n_runs)
    Parallel(n_jobs=njobs, verbose=10)(
        delayed(execute_run)(
            i_run,
            seeds[i_run],
            args,
            dataset_id=dataset,
            filepath=datapath,
            filesuffix=datasuffix,
        )
        for i_run in range(args.n_runs)
    )


def execute_run(
    i_run: int,
    rng: int,
    args: argparse.Namespace,
    dataset_id: str = "blobs",
    filepath="./datasets/",
    filesuffix="_ds18",
):
    """executes single training run

    Args:
        i_run (int): run id
        rng (int): seed for random number generators
        args (argparse.Namespace): parameters
        dataset_id (str, optional): which dataset to use (blobs or trees). Defaults to "blobs".
    """
    print("run {} / {}".format(str(i_run), str(args.n_runs)))

    # set random seed
    np.random.seed(rng)
    random.seed(rng)
    torch.manual_seed(rng)

    # create checkpoint dir
    run_name = "run_" + str(i_run)
    save_dir = Path("checkpoints") / args.save_dir / run_name
    # get (cuda) device
    args.device, _ = utils.nnet.get_device(args.cuda)
    args.verbose = False

    # get dataset
    if dataset_id == "blobs":
        dataset = utils.data.make_blobs_dataset(args)
    elif dataset_id == "trees":
        dataset = utils.data.make_trees_dataset(
            args, filepath=filepath, filesuffix=filesuffix
        )

    # instantiate logger, model and optimiser
    logger = hebbcl.logger.LoggerFactory.create(args, save_dir)
    model = hebbcl.model.ModelFactory.create(args)
    optim = hebbcl.trainer.Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    if dataset_id == "blobs":
        hebbcl.trainer.train_on_blobs(args, model, optim, dataset, logger)
    elif dataset_id == "trees":
        hebbcl.trainer.train_on_trees(args, model, optim, dataset, logger)

    # save results
    if args.save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.save(model)
