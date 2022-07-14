import numpy as np
import torch
import argparse

from typing import Dict


class Optimiser:
    """custom optimiser for SGD + Hebbian training updates"""

    def __init__(self, args: argparse.Namespace):
        """Constructor for optimiser

        Args:
            args (argparse.Namespace): training params specified in parameters.py
        """
        self.lrate_sgd = args.lrate_sgd
        self.lrate_hebb = args.lrate_hebb
        self.hebb_normaliser = args.hebb_normaliser
        self.perform_sgd = args.perform_sgd
        self.perform_hebb = args.perform_hebb
        self.gating = args.gating
        self.losstype = args.loss_funct
        self.n_features = args.n_features
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.ctx_twice = args.ctx_twice

    def step(self, model: torch.nn.Module, x_in: torch.Tensor, r_target: torch.Tensor):
        """a single training step, using procedure specified in args

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training inputs
            r_target (torch.Tensor): training targets
        """

        if self.perform_sgd is True:
            self._sgd_update(model, x_in, r_target)
        if self.perform_hebb is True:
            if self.n_layers == 1:
                if self.gating == "SLA":
                    self._sla_update(model, x_in)
                elif self.gating == "GHA":
                    self._gha_update(model, x_in)
                elif self.gating == "oja":
                    self._oja_update(model, x_in)
                elif self.gating == "oja_ctx":
                    self._oja_ctx_update(model, x_in)
            elif self.n_layers == 2:
                if self.gating == "oja_ctx":
                    self._oja_ctx_update_2hidden(model, x_in)
                elif self.gating == "oja":
                    self._oja_update_2hidden(model, x_in)
                else:
                    raise NotImplementedError(
                        "Only oja_ctx or oja supported for 2 layer net"
                    )

    def _sgd_update(
        self, model: torch.nn.Module, x_in: torch.Tensor, r_target: torch.Tensor
    ):
        """performs stochastic gradient descent

        Args:
            model (torch.nn.Module): neural network
            x_in (torch.Tensor): training input data
            r_target (torch.Tensor): training labels
        """
        y_ = model(x_in)
        # compute loss
        loss = self.loss_funct(r_target, y_)
        # get gradients
        loss.backward()
        # update weights
        with torch.no_grad():
            for theta in model.parameters():
                if theta.requires_grad:
                    theta -= theta.grad * self.lrate_sgd
            model.zero_grad()

    def _oja_update(self, model: torch.nn.Module, x_in: torch.Tensor):
        """applies Oja's rule to weights from context units to hidden units
        a vectorised implementation of slowoja_update

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """
        x_vec = x_in.repeat(self.n_hidden).reshape(-1, self.n_features)

        with torch.no_grad():
            y = torch.t(model.W_h) @ x_in
            y = y.repeat(self.n_features).reshape(self.n_features, -1).T
            dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h))
            model.W_h += dW.T
            model.zero_grad()

    def _oja_ctx_update(self, model: torch.nn.Module, x_in: torch.Tensor):
        """same as oja_update, but applied only to context units

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """
        x_in = x_in[-2:]
        x_vec = x_in.repeat(self.n_hidden).reshape(-1, 2)

        with torch.no_grad():
            y = torch.t(model.W_h[-2:, :]) @ x_in
            y = y.repeat(2).reshape(2, -1).T
            dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h[-2:, :]))
            model.W_h[-2:, :] += dW.T
            model.zero_grad()

    def _oja_ctx_update_2hidden(self, model: torch.nn.Module, x_in: torch.Tensor):
        """same as above, but for net with two hidden layers

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """
        x_in = x_in[-2:]
        x_vec = x_in.repeat(self.n_hidden).reshape(-1, 2)

        with torch.no_grad():
            y = torch.t(model.W_h1[-2:, :]) @ x_in
            y = y.repeat(2).reshape(2, -1).T
            dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h1[-2:, :]))
            model.W_h1[-2:, :] += dW.T
            model.zero_grad()

        if self.ctx_twice:  # apply hebb to second hidden layer
            with torch.no_grad():
                y = torch.t(model.W_h2[-2:, :]) @ x_in
                y = y.repeat(2).reshape(2, -1).T
                dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h2[-2:, :]))
                model.W_h2[-2:, :] += dW.T
                model.zero_grad()

    def _oja_update_2hidden(self, model: torch.nn.Module, x_in: torch.Tensor):
        """oja update applied to all units of first hidden layer in nnet with two hidden layers

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """
        x_vec = x_in.repeat(self.n_hidden).reshape(-1, self.n_features)

        with torch.no_grad():
            y = torch.t(model.W_h1) @ x_in
            y = y.repeat(self.n_features).reshape(self.n_features, -1).T
            dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h1))
            model.W_h1 += dW.T
            model.zero_grad()

        if self.ctx_twice:
            raise NotImplementedError(
                "oja on all units only supported for first hidden layer"
            )

    def _sla_update(self, model: torch.nn.Module, x_in: torch.Tensor):
        """applies subspace learning algorithm to input-to-hidden weights

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """

        x_in = x_in.reshape(self.n_features, 1)
        with torch.no_grad():
            Y = torch.t(model.W_h) @ x_in
            Y = Y.reshape(-1)
            x_in = x_in.reshape(-1)
            model.W_h += torch.t(
                (
                    torch.outer(Y, x_in)
                    - torch.outer(Y, model.W_h @ Y) / self.hebb_normaliser
                )
                * self.lrate_hebb
            )
            model.zero_grad()

    def _gha_update(self, model: torch.nn.Module, x_in: torch.Tensor):
        """applies generalised hebbian algorithm update to input-to-hidden weights

        Args:
            model (torch.nn.Module): neural network
            x_in (torch.Tensor): training data
        """
        x_in = torch.t(x_in)
        with torch.no_grad():
            Y = torch.t(model.W_h) @ x_in
            model.W_h += torch.t(
                (
                    torch.outer(Y, x_in)
                    - (torch.tril(torch.outer(Y, Y)) @ torch.t(model.W_h))
                    / self.hebb_normaliser
                )
                * self.lrate_hebb
            )
            model.zero_grad()

    def loss_funct(self, reward: torch.Tensor, y_hat: torch.Tensor) -> torch.float:
        """sets loss function, either negative reward or mse on model outputs

        Args:
            reward (torch.Tensor): reward
            y_hat (torch.Tensor): label

        Returns:
            torch.float: the computed loss
        """
        if self.losstype == "reward":
            # minimise -1*reward
            loss = -1 * torch.t(torch.sigmoid(y_hat)) @ reward
        elif self.losstype == "mse":
            loss = torch.mean(torch.pow(reward - y_hat, 2))
        elif self.losstype == "rew_on_sigmoid":
            loss = -1 * torch.t(y_hat) @ reward
        return loss


def train_on_blobs(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optim: Optimiser,
    data: Dict[str, np.array],
    logger,
):
    """trains a neural network on blobs task

    Args:
        args (argparse.Namespace): training parameters
        model (torch.nn.Module): feed forward neural network
        optim (Optimiser): optimiser that performs the training procedure
        data (Dict[str, np.array]): dictionary with training data
        logger (logger.LoggerFactory): a metric logger to keep track of training progress
    """

    # send data to gpu
    x_train = torch.from_numpy(data["x_train"]).float().to(args.device)
    y_train = torch.from_numpy(data["y_train"]).float().to(args.device)

    # test: create test sets
    x_a = torch.from_numpy(data["x_test_a"]).float().to(args.device)
    r_a = torch.from_numpy(data["y_test_a"]).float().to(args.device)

    x_b = torch.from_numpy(data["x_test_b"]).float().to(args.device)
    r_b = torch.from_numpy(data["y_test_b"]).float().to(args.device)

    x_both = (
        torch.from_numpy(np.concatenate((data["x_test_a"], data["x_test_b"]), axis=0))
        .float()
        .to(args.device)
    )
    r_both = (
        torch.from_numpy(np.concatenate((data["y_test_a"], data["y_test_b"]), axis=0))
        .float()
        .to(args.device)
    )

    f_both = torch.from_numpy(data["f_all"]).float().to(args.device)

    # log state of initialised model
    logger.log_init(model)
    logger.log_patterns(model, x_both)

    # loop over data and apply optimiser
    idces = np.arange(len(x_train))
    for ii, x, y in zip(idces, x_train, y_train):
        optim.step(model, x, y)
        if ii % args.log_interval == 0:

            logger.log_step(model, optim, x_a, x_b, x_both, r_a, r_b, r_both, f_both)
            if args.verbose:
                print(
                    "step {}, loss: task a {:.4f}, task b {:.4f} | acc: task a {:.4f}, task b {:.4f}".format(
                        str(ii),
                        logger.results["losses_1st"][-1],
                        logger.results["losses_2nd"][-1],
                        logger.results["acc_1st"][-1],
                        logger.results["acc_2nd"][-1],
                    )
                )
                print(
                    "... n_a: {:d} n_b: {:d}".format(
                        logger.results["n_only_a_regr"][-1],
                        logger.results["n_only_b_regr"][-1],
                    )
                )

    logger.log_patterns(model, x_both)
    print("done")


def train_on_trees(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optim: Optimiser,
    data: Dict[str, np.array],
    logger,
):
    """trains a neural network on trees task

    Args:
        args (argparse.Namespace): training parameters
        model (torch.nn.Module): feed forward neural network
        optim (Optimiser): optimiser that performs the training procedure
        data (Dict[str, np.array]): dictionary with training data
        logger (logger.LoggerFactory): a metric logger to keep track of training progress
    """

    # send data to gpu
    x_train = torch.from_numpy(data["x_train"]).float().to(args.device)
    y_train = torch.from_numpy(data["y_train"]).float().to(args.device)

    # test: create test sets
    x_a = torch.from_numpy(data["x_test_a"]).float().to(args.device)
    r_a = torch.from_numpy(data["y_test_a"]).float().to(args.device)

    x_b = torch.from_numpy(data["x_test_b"]).float().to(args.device)
    r_b = torch.from_numpy(data["y_test_b"]).float().to(args.device)

    x_both = (
        torch.from_numpy(np.concatenate((data["x_test_a"], data["x_test_b"]), axis=0))
        .float()
        .to(args.device)
    )
    r_both = (
        torch.from_numpy(np.concatenate((data["y_test_a"], data["y_test_b"]), axis=0))
        .float()
        .to(args.device)
    )

    x_pattern = (
        torch.from_numpy(
            np.concatenate((data["x_test_a"][:25, :], data["x_test_b"][:25, :]), axis=0)
        )
        .float()
        .to(args.device)
    )

    f_pattern = (
        torch.from_numpy(
            np.concatenate((data["f_test_a"][:25, :], data["f_test_b"][:25, :]), axis=0)
        )
        .float()
        .to(args.device)
    )

    # log state of initialised model
    logger.log_init(model)
    logger.log_patterns(model, x_pattern)

    # loop over data and apply optimiser
    idces = np.arange(len(x_train))
    for ii, x, y in zip(idces, x_train, y_train):
        optim.step(model, x, y)
        if ii % args.log_interval == 0:

            logger.log_step(
                model,
                optim=optim,
                x_a=x_a,
                x_b=x_b,
                x_both=x_both,
                x_pattern=x_pattern,
                f_pattern=f_pattern,
                r_a=r_a,
                r_b=r_b,
                r_both=r_both,
            )
            if args.verbose:
                print(
                    "step {}, loss: task a {:.4f}, task b {:.4f} | acc: task a {:.4f}, task b {:.4f}".format(
                        str(ii),
                        logger.results["losses_1st"][-1],
                        logger.results["losses_2nd"][-1],
                        logger.results["acc_1st"][-1],
                        logger.results["acc_2nd"][-1],
                    )
                )
                print(
                    "...1st hidden: n_a: {:d} n_b: {:d}".format(
                        logger.results["n_only_a_regr"][-1],
                        logger.results["n_only_b_regr"][-1],
                    )
                )

                if args.n_layers > 1:
                    print(
                        "... 2nd hidden: n_a: {:d} n_b: {:d}".format(
                            logger.results["n_only_a_regr2"][-1],
                            logger.results["n_only_b_regr2"][-1],
                        )
                    )

    logger.log_patterns(model, x_pattern)
    print("done")
