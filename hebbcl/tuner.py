import torch
import numpy as np
import random
import argparse
import utils
import hebbcl
import ray
from ray import tune
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest import SearchAlgorithm
from ray.tune.suggest.suggestion import Searcher
from ray.tune.schedulers import HyperBandForBOHB, ASHAScheduler, TrialScheduler
from typing import Union


class HPOTuner(object):
    def __init__(
        self, args: argparse.Namespace, time_budget: int = 100, metric: str = "loss"
    ):
        """hyperparameter optimisation for nnets

        Args:
            args (ArgumentParser): collection of neural network training parameters
            time_budget (int, optional): time budget allocated to the fitting process (in seconds). Defaults to 100.
            metric (str, optional): metric to optimise, can be "acc" or "loss". Defaults to "loss".
        """

        self.metric = self._set_metric(metric)
        self.mode = self._set_mode(metric)

        self.time_budget = time_budget
        self.args = args

        self.best_cfg = None
        self.results = None

        if self.args.hpo_fixedseed:
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

        ray.shutdown()
        ray.init(
            runtime_env={"working_dir": "../ray_tune/", "py_modules": [utils, hebbcl]}
        )

    def tune(self, time_budget: int = None, n_samples: int = None):
        """runs the tuner. time budget and n trials set in constructor can be overwritten here

        Args:
            time_budget (int, optional): time in seconds allocated to the fitting proces. Defaults to 100.
            n_samples (int, optional): number of trials. Defaults to 100.
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
            resources_per_trial={"cpu": 1, "gpu": 0},
            verbose=1,
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
        data = utils.data.make_dataset(self.args)

        # instantiate model and hebbcl.trainer.Optimiser
        if self.args.gating == "manual":
            model = hebbcl.model.Gatednet(self.args)
        else:
            model = hebbcl.model.Nnet(self.args)
        optim = hebbcl.trainer.Optimiser(self.args)

        # send model and data to device
        model = model.to(self.args.device)
        x_train = torch.from_numpy(data["x_train"]).float().to(self.args.device)
        y_train = torch.from_numpy(data["y_train"]).float().to(self.args.device)

        x_both = (
            torch.from_numpy(
                np.concatenate((data["x_task_a"], data["x_task_b"]), axis=0)
            )
            .float()
            .to(self.args.device)
        )
        r_both = (
            torch.from_numpy(
                np.concatenate((data["y_task_a"], data["y_task_b"]), axis=0)
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
                "n_episodes": tune.choice([200, 500, 1000]),
            }

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
