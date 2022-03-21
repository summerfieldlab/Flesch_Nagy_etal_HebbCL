import torch
import numpy as np
from argparse import ArgumentParser
from utils.data import make_dataset
from utils.nnet import from_gpu
from utils.eval import compute_accuracy

from hebbcl.model import Nnet, Gatednet
from hebbcl.trainer import Optimiser
from hebbcl.parameters import parser
from ray import tune


def train_nnet_with_ray(config: dict, args: ArgumentParser) -> dict:
    """trains neural network model

    Args:
        config (dict): dictionary with raytune config
        args (ArgumentParser): params for nnet

    Returns:
        dict: raytune results
    """

    # get dataset
    data = make_dataset(args)

    # replace args with ray tune config
    if args.gating == "SLA":
        args.lrate_sgd = config["lrate_sgd"]
        args.perform_hebb = True if config["sla"] == 1 else False
        args.hebb_normaliser = config["normaliser"]
        args.lrate_hebb = config["lrate_hebb"]

    elif args.gating == "manual":
        args.lrate_sgd = config["lrate_sgd"]
        args.n_episodes = config["n_episodes"]
        args.weight_init = config["weight_init"]

    # instantiate model and optimiser
    if args.gating == "manual":
        model = Gatednet(args)
    else:
        model = Nnet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # send data to gpu
    x_train = torch.from_numpy(data["x_train"]).float().to(args.device)
    y_train = torch.from_numpy(data["y_train"]).float().to(args.device)

    x_both = (
        torch.from_numpy(np.concatenate((data["x_task_a"], data["x_task_b"]), axis=0))
        .float()
        .to(args.device)
    )
    r_both = (
        torch.from_numpy(np.concatenate((data["y_task_a"], data["y_task_b"]), axis=0))
        .float()
        .to(args.device)
    )

    # loop over data and apply optimiser
    idces = np.arange(len(x_train))
    for ii, x, y in zip(idces, x_train, y_train):
        optim.step(model, x, y)

        if ii % args.log_interval == 0:
            # obtain loss/acc metrics
            loss_both = from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0]
            acc_both = compute_accuracy(r_both, model(x_both))

            # send metrics to ray tune
            tune.report(mean_loss=loss_both, mean_acc=acc_both)


def run_raytune(config: dict, args: ArgumentParser) -> dict:
    """performs raytun HPO

    Args:
        config (dict): dicitonary with raytune hyperopt searchspace
        args (ArgumentParser): nnet params

    Returns:
        dict: results
    """
    # run ray tune
    analysis = tune.run(
        train_nnet_with_ray,
        config=config,
        num_samples=100,
        metric="mean_loss",
        mode="min",
        resources_per_trial={"cpu": 1, "gpu": 0},
    )
    best_cfg = analysis.get_best_config(metric="mean_loss", mode="min")
    print("Best config: ", best_cfg)

    # results as dataframe
    df = analysis.results_df
    results = {"df": df, "best": best_cfg}
    return results
