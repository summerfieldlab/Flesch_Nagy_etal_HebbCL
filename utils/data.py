import argparse
import pickle
from PIL import Image
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from scipy.stats import multivariate_normal


def resize_images(
    filename: str = None, filepath: str = "../datasets/", size: tuple = (24, 24)
):
    """crops and resizes rgb images in pickle file

    Args:
        filename (str, optional): name of dataset to load. Defaults to None.
        filepath (str, optional): path to dataset. Defaults to "../datasets".
        size (tuple, optional): new size in pixels. Defaults to (24, 24).
    """

    with open(filepath + filename + ".pkl", "rb") as f:
        data = pickle.load(f)

    x_rs = np.array(
        list(
            map(
                lambda x: np.asarray(
                    Image.fromarray(x.reshape((96, 96, 3))).resize(size)
                ).flatten(),
                data["images"],
            )
        )
    )
    data["images"] = x_rs

    with open(filepath + filename + "_ds" + str(size[0]) + ".pkl", "wb") as f:
        pickle.dump(data, f)


def crop_resize_images(
    filename: str = None, filepath: str = "../datasets/", size: tuple = (18, 18)
):
    """crops and resizes rgb images in pickle file

    Args:
        filename (str, optional): name of dataset to load. Defaults to None.
        filepath (str, optional): path to dataset. Defaults to "../datasets".
        size (tuple, optional): new size in pixels. Defaults to (18, 18).
    """

    with open(filepath + filename + ".pkl", "rb") as f:
        data = pickle.load(f)

    # old and new width/height to crop out contexts
    w = 96
    h = 96
    nw = 72
    nh = 72

    x_rs = np.array(
        list(
            map(
                lambda x: np.asarray(
                    Image.fromarray(x.reshape((96, 96, 3)))
                    .crop(((w - nw) // 2, (h - nh) // 2, (w + nw) // 2, (h + nh) // 2))
                    .resize(size)
                ).flatten(),
                data["images"],
            )
        )
    )
    data["images"] = x_rs

    with open(
        filepath + filename.replace("_withgarden", "") + "_ds" + str(size[0]) + ".pkl",
        "wb",
    ) as f:
        pickle.dump(data, f)


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


def make_trees_block(
    context: str,
    do_shuffle: bool = False,
    c_scaling: int = 1,
    filepath: str = "../datasets/",
    whichset: str = "training",
    exemplar: int = 0,
    filesuffix: str = "_ds18",
) -> Tuple[np.array, np.array, np.array]:
    """generates a training block with stimuli from trees dataset

    Args:
        context (str): which task task_a (north) or task_b (south).
        do_shuffle (bool, optional): shuffle trials?. Defaults to False.
        c_scaling (int, optional): context offset. Defaults to 1.
        filepath( str, optional): path to datasets. Defaults to "../datasets/"
        whichset (str, optional): "training" or "test". Defaults to "training".
        exemplar (int, optional): which exemplar, int in [0,199]. Defaults to 0.

    Returns:
        Tuple[np.array, np.array, np.array]: input images, rewards, feature values
    """
    assert context in ["task_a", "task_b"]
    assert whichset in ["training", "test"]
    context = "north" if context == "task_a" else "south"

    # load appropriate dataset
    with open(
        filepath + whichset + "_data_" + context + filesuffix + ".pkl", "rb"
    ) as f:
        data = pickle.load(f)

    # pull out data belonging to specific exemplar
    idces = data["exemplars"] == exemplar
    for k in data.keys():
        data[k] = data[k][
            idces,
        ]

    # reformat and add context signal
    contexts = np.zeros((len(data["images"]), 2))
    contexts[:, data["contexts"] - 1] = 1 * c_scaling
    data["images"] = np.concatenate(
        (data["images"].astype("float") / 255, contexts), axis=1
    )

    if do_shuffle:
        idces = shuffle(np.arange(len(data["images"])))
        data["images"] = data["images"][idces, :]
        data["rewards"] = data["rewards"][idces, :]
        data["branchiness"] = data["branchiness"][
            idces,
        ]
        data["leafiness"] = data["leafiness"][
            idces,
        ]
    # return only required stuff
    return (
        data["images"],
        data["rewards"][:, 0].reshape((-1, 1)),
        np.stack((data["branchiness"], data["leafiness"]), axis=1),
    )


def make_blobs_block(
    context: str, do_shuffle: bool, c_scaling: int = 1
) -> Tuple[np.array, np.array, np.array]:
    """generates a training block with stimuli from blobs dataset

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
    l = l.flatten()  # noqa E741
    r_s, r_n = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5), np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    ii_sub = 1
    blobs = np.empty((25, n_units))
    for ii in range(0, 25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii], xy_sigma=0.08, n=resolution)
        blob = blob / np.max(blob)
        ii_sub += 1
        blobs[ii, :] = blob.flatten()

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


def make_trees_blocks(
    do_shuffle: bool = False,
    whichtask: str = "task_a",
    whichset: str = "training",
    n_blocks: int = 10,
    c_scaling: int = 1,
    n_max: int = 199,
    filepath: str = "../datasets/",
    filesuffix: str = "_ds18",
) -> Tuple[np.array, np.array, np.array]:
    """todo

    Args:
        do_shuffle (bool, optional): todo. Defaults to False.
        whichtask (str, optional): todo. Defaults to "task_a".
        whichset (str, optional): todo. Defaults to "training".
        n_blocks (int, optional): todo. Defaults to 10.
        c_scaling (int, optional): todo. Defaults to 1.
        filepath( str, optional): path to datasets. Defaults to "../datasets/"

    Returns:
        Tuple[np.array, np.array, np.array]: todo
    """
    episode_idces = np.random.permutation(n_max)[:n_blocks]
    x_a, y_a, f_a = None, None, None
    for e in episode_idces:
        xe, ye, fe = make_trees_block(
            whichtask,
            do_shuffle=do_shuffle,
            c_scaling=c_scaling,
            exemplar=e,
            whichset=whichset,
            filepath=filepath,
            filesuffix=filesuffix,
        )
        x_a = np.vstack((x_a, xe)) if x_a is not None else xe
        y_a = np.vstack((y_a, ye)) if y_a is not None else ye
        f_a = np.vstack((f_a, fe)) if f_a is not None else fe
    return x_a, y_a, f_a


def make_trees_dataset(
    args: argparse.Namespace, filepath: str = "../datasets/", filesuffix="_ds18"
) -> dict:
    """todo

    Args:
        args (argparse.Namespace): todo
        filepath( str, optional): path to datasets. Defaults to "../datasets/"

    Returns:
        dict: todo
    """
    # training dataset:
    # for n_episodes: sample exemplar indices, make block
    x_a, y_a, f_a = make_trees_blocks(
        whichtask="task_a",
        whichset="training",
        do_shuffle=True,
        c_scaling=args.ctx_scaling,
        n_blocks=args.n_episodes // 2,
        n_max=399,
        filepath=filepath,
        filesuffix=filesuffix,
    )
    x_b, y_b, f_b = make_trees_blocks(
        whichtask="task_b",
        whichset="training",
        do_shuffle=True,
        c_scaling=args.ctx_scaling,
        n_blocks=args.n_episodes // 2,
        n_max=399,
        filepath=filepath,
        filesuffix=filesuffix,
    )

    data = {}
    if args.training_schedule == "blocked":
        shuff_idces = np.random.permutation(len(x_a))
        data["x_train"] = np.vstack(
            (
                x_a[
                    shuff_idces,
                ],
                x_b[
                    shuff_idces,
                ],
            )
        )
        data["y_train"] = np.vstack(
            (
                y_a[
                    shuff_idces,
                ],
                y_b[
                    shuff_idces,
                ],
            )
        )
        data["f_train"] = np.vstack(
            (
                f_a[
                    shuff_idces,
                ],
                f_b[
                    shuff_idces,
                ],
            )
        )
    elif args.training_schedule == "interleaved":
        shuff_idces = np.random.permutation(len(x_a) * 2)
        data["x_train"] = np.vstack((x_a, x_b))[
            shuff_idces,
        ]
        data["y_train"] = np.vstack((y_a, y_b))[
            shuff_idces,
        ]
        data["f_train"] = np.vstack((f_a, f_b))[
            shuff_idces,
        ]

    # now get test datasets (for task a and b)
    data["x_test_a"], data["y_test_a"], data["f_test_a"] = make_trees_blocks(
        whichtask="task_a",
        whichset="test",
        do_shuffle=False,
        c_scaling=args.ctx_scaling,
        n_blocks=10,
        n_max=199,
        filepath=filepath,
        filesuffix=filesuffix,
    )
    data["x_test_b"], data["y_test_b"], data["f_test_b"] = make_trees_blocks(
        whichtask="task_b",
        whichset="test",
        do_shuffle=False,
        c_scaling=args.ctx_scaling,
        n_blocks=10,
        n_max=199,
        filepath=filepath,
        filesuffix=filesuffix,
    )

    if args.centering is True:
        sc = StandardScaler(with_std=False)
        data["x_train"] = sc.fit_transform(data["x_train"])
        data["x_test_a"] = sc.transform(data["x_test_a"])
        data["x_test_b"] = sc.transform(data["x_test_b"])

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


def make_blobs_dataset(args: argparse.Namespace) -> dict:
    """makes dataset for experiment with blobs stimuli

    Args:
        args (argparse.Namespace): training parameters

    Returns:
        dict: inputs and labels for training and test phase
    """

    random_state = np.random.randint(999)

    x_test_a, y_test_a, f_test_a = make_blobs_block("task_a", 0, args.ctx_scaling)
    y_test_a = y_test_a[:, np.newaxis]
    l_test_a = (y_test_a > 0).astype("int")

    x_test_b, y_test_b, f_test_b = make_blobs_block("task_b", 0, args.ctx_scaling)
    y_test_b = y_test_b[:, np.newaxis]
    l_test_b = (y_test_b > 0).astype("int")

    if args.ctx_weights is True:
        x_test_a[:, :25] /= np.linalg.norm(x_test_a[:, :25])
        x_test_a[:, 25:] /= np.linalg.norm(x_test_a[:, 25:])
        x_test_b[:, :25] /= np.linalg.norm(x_test_b[:, :25])
        x_test_b[:, 25:] /= np.linalg.norm(x_test_b[:, 25:])

    x_in = np.concatenate((x_test_a, x_test_b), axis=0)
    y_rew = np.concatenate((y_test_a, y_test_b), axis=0)
    y_true = np.concatenate((l_test_a, l_test_b), axis=0)
    f_all = np.concatenate((f_test_a, f_test_b), axis=0)

    # define datasets (duplicates for shuffling)
    data = {}
    data["x_test_a"] = x_test_a
    data["y_test_a"] = y_test_a
    data["l_test_a"] = l_test_a
    data["f_test_a"] = f_test_a

    data["x_test_b"] = x_test_b
    data["y_test_b"] = y_test_b
    data["l_test_b"] = l_test_b
    data["f_test_b"] = f_test_b

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
                            shuffle(data["x_test_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["x_test_b"], random_state=i + random_state)
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
                            shuffle(data["y_test_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["y_test_b"], random_state=i + random_state)
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
                            shuffle(data["l_test_a"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["l_test_b"], random_state=i + random_state)
                            for i in range(args.n_episodes)
                        ]
                    )
                ),
            )
        )
    else:
        print("Unknown training schedule parameter")

    if args.centering is True:
        sc = StandardScaler(with_std=False)
        data["x_train"] = sc.fit_transform(data["x_train"])
        data["x_test_a"] = sc.transform(data["x_test_a"])
        data["x_test_b"] = sc.transform(data["x_test_b"])
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
