import torch
import numpy as np
from argparse import ArgumentParser
import utils
import hebbcl
import ray
from ray import tune


class HPOTuner(object):
    def __init__(
        self, args: ArgumentParser, time_budget: int = 100, metric: str = "loss"
    ):

        self.metric = self._set_metric(metric)
        self.mode = self._set_mode(metric)

        self.time_budget = time_budget
        self.args = args

        self.best_cfg = None
        self.results = None

        ray.shutdown()
        ray.init(
            runtime_env={"working_dir": "../ray_tune/", "py_modules": [utils, hebbcl]}
        )

    def tune(self, time_budget=None, n_samples=None):
        """tunes trainable"""
        # run ray tune
        analysis = tune.run(
            self._trainable,
            config=self._get_config(),
            time_budget_s=time_budget if time_budget else self.time_budget,
            num_samples=n_samples if n_samples else 100,
            metric=self.metric,
            mode=self.mode,
            resources_per_trial={"cpu": 1, "gpu": 0},
        )
        self.best_cfg = analysis.get_best_config(metric=self.metric, mode=self.mode)
        
        # results as dataframe
        self.results = analysis.results_df
        
    def _trainable(self, config: dict):
        """function to optimise"""
        self.args.device, _ = utils.nnet.get_device(self.args.cuda)
        print(self.args.device)
        # get dataset
        data = utils.data.make_dataset(self.args)
        # replace self.args with ray tune config
        for k, v in config.items():
            setattr(self.args, k, v)

        # instantiate model and hebbcl.trainer.Optimiser
        if self.args.gating == "manual":
            model = hebbcl.model.Gatednet(self.args)
        else:
            model = hebbcl.model.Nnet(self.args)
        optim = hebbcl.trainer.Optimiser(self.args)

        # send model to GPU
        model = model.to(self.args.device)

        # send data to gpu
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

    def _get_config(self):
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

        return config

    def _set_metric(self, metric: str):
        if metric == "loss":
            return "mean_loss"
        elif metric == "acc":
            return "mean_acc"
        else:
            raise ValueError("Invalid metric provided (choose 'loss' or 'acc'")

    def _set_mode(self, metric: str):
        if metric == "loss":
            return "min"
        elif metric == "acc":
            return "max"
        else:
            raise ValueError("Invalid metric provided (choose 'loss' or 'acc'")
