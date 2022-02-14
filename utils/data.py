import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from scipy.stats import multivariate_normal


def gen2Dgauss(x_mu=0.0, y_mu=0.0, xy_sigma=0.1, n=20) -> np.array:
    """generates a two-dimensional Gaussian blob

    Args:
        x_mu (float, optional): x location of mean. Defaults to 0.0.
        y_mu (float, optional): y location of mean. Defaults to 0.0.
        xy_sigma (float, optional): diagonal entries of covariance matrix (isotropic). Defaults to 0.1.
        n (int, optional): resolution of meshgrid. Defaults to 20.

    Returns:
        np.array: z-values of Gaussian blob
    """
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    gausspdf = multivariate_normal([x_mu, y_mu], [[xy_sigma, 0], [0, xy_sigma]])
    x_in = np.empty(xx.shape + (2,))
    x_in[:, :, 0] = xx
    x_in[:, :, 1] = yy
    return gausspdf.pdf(x_in)


# def mk_block(context: str, do_shuffle: bool) -> Tuple[np.array, np.array, np.array]:
#     """generates a training block

#     Args:
#         context (str): task a or task b
#         do_shuffle (bool): shuffle trials

#     Returns:
#         Tuple: inputs, rewards and feature values
#     """
#     resolution = 5
#     n_units = resolution ** 2
#     l, b = np.meshgrid(np.linspace(0.2, 0.8, 5), np.linspace(0.2, 0.8, 5))
#     b = b.flatten()
#     l = l.flatten()
#     r_n, r_s = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
#     r_s = r_s.flatten()
#     r_n = r_n.flatten()
#     val_l, val_b = np.meshgrid(np.linspace(1, 5, 5), np.linspace(1, 5, 5))
#     val_b = val_b.flatten()
#     val_l = val_l.flatten()

#     ii_sub = 1
#     blobs = np.empty((25, n_units))
#     for ii in range(0, 25):
#         blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii], xy_sigma=0.08, n=resolution)
#         blob = blob / np.max(blob)
#         ii_sub += 1
#         blobs[ii, :] = blob.flatten()
#     x1 = blobs
#     if context == "task_a":
#         reward = r_n
#     elif context == "task_b":
#         reward = r_s

#     feature_vals = np.vstack((val_b, val_l)).T
#     if do_shuffle:
#         ii_shuff = np.random.permutation(25)
#         x1 = x1[ii_shuff, :]
#         feature_vals = feature_vals[ii_shuff, :]
#         reward = reward[ii_shuff]
#     return x1, reward, feature_vals


def mk_block_wctx(
    context: str, do_shuffle: bool, c_scaling=1
) -> Tuple[np.array, np.array, np.array]:
    """generates a training block

    Args:
        context (str): task a or task b
        do_shuffle (bool): shuffle trials
        c_scaling (int, optional): scaling of context dimension. Defaults to 1.

    Returns:
        Tuple[np.array, np.array, np.array]: inputs, rewards and feature values
    """
    resolution = 5
    n_units = resolution ** 2
    l, b = np.meshgrid(np.linspace(0.2, 0.8, 5), np.linspace(0.2, 0.8, 5))
    b = b.flatten()
    l = l.flatten()
    r_s, r_n = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5), np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    # plt.figure()
    ii_sub = 1
    blobs = np.empty((25, n_units))
    for ii in range(0, 25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii], xy_sigma=0.08, n=resolution)
        blob = blob / np.max(blob)
        ii_sub += 1
        blobs[ii, :] = blob.flatten()
    x = blobs
    if context == "task_a":
        x1 = np.append(blobs, c_scaling * np.ones((blobs.shape[0], 1)), axis=1)
        x1 = np.append(x1, np.zeros((blobs.shape[0], 1)), axis=1)
        reward = r_n
    elif context == "task_b":
        x1 = np.append(blobs, np.zeros((blobs.shape[0], 1)), axis=1)
        x1 = np.append(x1, c_scaling * np.ones((blobs.shape[0], 1)), axis=1)
        reward = r_s

    feature_vals = np.vstack((val_b, val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff, :]
        feature_vals = feature_vals[ii_shuff, :]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals


def make_dataset(args: argparse.ArgumentParser) -> dict:
    """makes dataset for experiment

    Args:
        args (argparse.ArgumentParser): training parameters

    Returns:
        dict: inputs and labels for training and test phase
    """

    random_state = np.random.randint(999)

    x_task_a, y_task_a, f_task_a = mk_block_wctx("task_a", 0, args.ctx_scaling)
    y_task_a = y_task_a[:, np.newaxis]
    l_task_a = (y_task_a > 0).astype("int")

    x_task_b, y_task_b, f_task_b = mk_block_wctx("task_b", 0, args.ctx_scaling)
    y_task_b = y_task_b[:, np.newaxis]
    l_task_b = (y_task_b > 0).astype("int")

    if args.ctx_weights == True:
        x_task_a[:, :25] /= np.linalg.norm(x_task_a[:, :25])
        x_task_a[:, 25:] /= np.linalg.norm(x_task_a[:, 25:])
        x_task_b[:, :25] /= np.linalg.norm(x_task_b[:, :25])
        x_task_b[:, 25:] /= np.linalg.norm(x_task_b[:, 25:])

    x_in = np.concatenate((x_task_a, x_task_b), axis=0)
    y_rew = np.concatenate((y_task_a, y_task_b), axis=0)
    y_true = np.concatenate((l_task_a, l_task_b), axis=0)
    f_all = np.concatenate((f_task_a, f_task_b), axis=0)

    # define datasets (duplicates for shuffling)
    data = {}
    data["x_task_a"] = x_task_a
    data["y_task_a"] = y_task_a
    data["l_task_a"] = l_task_a
    data["f_task_a"] = f_task_a

    data["x_task_b"] = x_task_b
    data["y_task_b"] = y_task_b
    data["l_task_b"] = l_task_b
    data["f_task_b"] = f_task_b

    data["x_all"] = x_in
    data["y_all"] = y_rew
    data["l_all"] = y_true
    data["f_all"] = f_all

    if args.training_schedule == "interleaved":
        data["x_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["x_all"], random_state=i + random_state)
                    for i in range(args.n_episodes)
                ]
            )
        )
        data["y_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["y_all"], random_state=i + random_state)
                    for i in range(args.n_episodes)
                ]
            )
        )
        data["l_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["l_all"], random_state=i + random_state)
                    for i in range(args.n_episodes)
                ]
            )
        )
    elif args.training_schedule == "blocked":
        data["x_train"] = np.vstack(
            (
                np.vstack(
                    tuple(
                        [
                            shuffle(data["x_task_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["x_task_b"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
            )
        )
        data["y_train"] = np.vstack(
            (
                np.vstack(
                    tuple(
                        [
                            shuffle(data["y_task_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["y_task_b"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
            )
        )
        data["l_train"] = np.vstack(
            (
                np.vstack(
                    tuple(
                        [
                            shuffle(data["l_task_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["l_task_b"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
            )
        )
    else:
        print("Unknown training schedule parameter")

    if args.centering == True:
        sc = StandardScaler(with_std=False)
        data["x_train"] = sc.fit_transform(data["x_train"])
        # data["x_task_a"] = sc.transform(data["x_task_a"])
        # data["x_task_b"] = sc.transform(data["x_task_b"])
        # x_in = StandardScaler(with_std=False).fit_transform(x_in)
        if args.training_schedule == "blocked":
            # remove info about 2nd task during training on 1st task
            data["x_train"][data["x_train"][:, -2] > 0, -1] = 0

    if args.ctx_avg:
        if args.ctx_avg_type == "sma":
            data["x_train"][:, -2] = (
                pd.Series(data["x_train"][:, -2])
                .rolling(window=args.ctx_avg_window, min_periods=1)
                .mean()
            )
            data["x_train"][:, -1] = (
                pd.Series(data["x_train"][:, -1])
                .rolling(window=args.ctx_avg_window, min_periods=1)
                .mean()
            )
        elif args.ctx_avg_type == "ema":
            data["x_train"][:, -2] = (
                pd.Series(data["x_train"][:, -2])
                .ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1)
                .mean()
            )
            data["x_train"][:, -1] = (
                pd.Series(data["x_train"][:, -1])
                .ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1)
                .mean()
            )

    return data
