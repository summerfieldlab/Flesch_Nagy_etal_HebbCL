import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from utils.nnet import from_gpu
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore, multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd


def make_dmat(features):
    """creates design matrix for task selectivity analysis

    Args:
        features (np array): n-x-2 matrix of feature values

    Returns:
        dmat: design matrix with z-scored predictors, one per feature & task
    """
    dmat = np.zeros((50, 4))
    # irrel in task a
    dmat[:25, 0] = features[:25, 0]
    # rel in task a
    dmat[:25, 1] = features[:25, 1]
    # rel in task b
    dmat[25:, 2] = features[25:, 0]
    # irrel in task b
    dmat[25:, 3] = features[25:, 1]
    return zscore(dmat, axis=0, ddof=1)


def compute_congruency_acc(cmat, cmat_true):
    """computes accuracy on congruent and incongruent trials

    Args:
        cmat (np array): choices
        cmat_true (np array): ground truth category labels

    Returns:
        int: accuracies on congruent and incongruent trials
    """
    c = (cmat > 0.5) == (cmat_true > 0.5)
    acc_congruent = (np.mean(c[:2, :2]) + np.mean(c[3:, 3:])) / 2
    acc_incongruent = (np.mean(c[:2, 3:]) + np.mean(c[3:, :2])) / 2
    return acc_congruent, acc_incongruent


def gen_behav_models():
    modelrdms = np.empty((2, 50, 50))
    cmats = []
    # factorised model:
    a, b = np.meshgrid([0, 0, 0.5, 1, 1], [0, 0, 0.5, 1, 1])

    x = np.concatenate((b.flatten(), a.flatten()), axis=0)[:, np.newaxis]
    cmats.append(x)
    modelrdms[0, :, :] = squareform(pdist(x))

    # diagonal model
    x1 = np.ones((5, 5))
    x1 = np.triu(x1)
    x1[np.diag_indices(5)] = 0.5
    x1 = np.flipud(x1)
    x = np.concatenate((x1.flatten(), x1.flatten()), axis=0)[:, np.newaxis]
    cmats.append(x)
    modelrdms[1, :, :] = squareform(pdist(x))

    rdm1 = modelrdms[0, :, :]
    rdm2 = modelrdms[1, :, :]
    cmats = np.array(cmats).reshape((2, 2, 5, 5))  # model, task, d1, d2
    dmat = np.array(
        [
            zscore(rdm1[np.tril_indices(50, k=-1)].flatten()),
            zscore(rdm2[np.tril_indices(50, k=-1)].flatten()),
        ]
    ).T
    return modelrdms, dmat, cmats


def rotate_axes(x, y, theta):
    # theta is in degrees
    theta_rad = theta * (math.pi / 180)  # convert to radians
    x_new = x * math.cos(theta_rad) + y * math.sin(theta_rad)
    y_new = -x * math.sin(theta_rad) + y * math.cos(theta_rad)
    return x_new, y_new


def rotate(X, theta, axis="x"):
    """Rotate multidimensional array `X` `theta` degrees around axis `axis`"""
    theta = theta * (math.pi / 180)  # convert to radians
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.dot(X, np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]]))
    elif axis == "y":
        return np.dot(X, np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]))
    elif axis == "z":
        return np.dot(
            X,
            np.array(
                [
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1.0],
                ]
            ),
        )


def compute_accuracy(y, y_):
    """
    accuracy for this experiment is defined as matching signs (>= and < 0) for outputs and targets
    The category boundary trials are neglected.
    """
    valid_targets = y != 0
    outputs = y_ > 0
    targets = y > 0
    return from_gpu(
        torch.mean((outputs[valid_targets] == targets[valid_targets]).float())
    ).ravel()[0]


def compute_sparsity_stats(yout):
    """
    computes task-specificity of activity patterns
    """
    # yout is n_units x n_trials
    # 1. average within contexts
    assert yout.shape == (100, 50)
    x = np.vstack((np.mean(yout[:, 0:25], 1).T, np.mean(yout[:, 25:-1], 1).T))
    # should yield a 2xn_hidden vector
    # now count n dead units (i.e. return 0 in both tasks)
    n_dead = np.sum(~np.any(x, axis=0))
    # now count number of local units in total (only active in one task)
    n_local = np.sum(~(np.all(x, axis=0)) & np.any(x, axis=0))
    # count units only active in task a
    n_only_A = np.sum(np.all(np.vstack((x[0, :] > 0, x[1, :] == 0)), axis=0))
    # count units only active in task b
    n_only_B = np.sum(np.all(np.vstack((x[0, :] == 0, x[1, :] > 0)), axis=0))
    # compute dot product of hiden layer activations
    h_dotprod = np.dot(x[0, :], x[1, :].T)
    # return all
    return n_dead, n_local, n_only_A, n_only_B, h_dotprod


def mse(y_, y):
    """
    computes mean squared error between targets and outputs
    """
    return 0.5 * np.linalg.norm(y_ - y, 2) ** 2


def compute_relchange(w0, wt):
    """
    computes relative change of norm of weights
    inputs:
    - w0: weights at initialisation
    - wt: weights at time t.
    output:
    - (norm(wt)-norm(w0))/norm(w0)
    """
    return (
        np.linalg.norm(wt.flatten()) - np.linalg.norm(w0.flatten())
    ) / np.linalg.norm(w0.flatten())


def gen2Dgauss(x_mu=0.0, y_mu=0.0, xy_sigma=0.1, n=20):
    """
    generates two-dimensional gaussian blob
    """
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    gausspdf = multivariate_normal([x_mu, y_mu], [[xy_sigma, 0], [0, xy_sigma]])
    x_in = np.empty(xx.shape + (2,))
    x_in[:, :, 0] = xx
    x_in[:, :, 1] = yy
    return gausspdf.pdf(x_in)


def mk_block_wctx(context, do_shuffle, c_scaling=1):
    """
    generates block of experiment
    Input:
      - task  : 'task_a' or 'task_b'
      - do_shuffle: True or False, shuffles  values
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


def make_dataset(
    ctx_scaling=1,
    training_schedule="blocked",
    n_episodes=10,
    ctx_avg=True,
    ctx_avg_window=10,
    centering=True,
):
    """
    makes dataset for experiment
    """

    random_state = np.random.randint(999)

    x_task_a, y_task_a, f_task_a = mk_block_wctx("task_a", 0, ctx_scaling)
    y_task_a = y_task_a[:, np.newaxis]
    l_task_a = (y_task_a > 0).astype("int")

    x_task_b, y_task_b, f_task_b = mk_block_wctx("task_b", 0, ctx_scaling)
    y_task_b = y_task_b[:, np.newaxis]
    l_task_b = (y_task_b > 0).astype("int")

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

    if training_schedule == "interleaved":
        data["x_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["x_all"], random_state=i + random_state)
                    for i in range(n_episodes)
                ]
            )
        )
        data["y_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["y_all"], random_state=i + random_state)
                    for i in range(n_episodes)
                ]
            )
        )
        data["l_train"] = np.vstack(
            tuple(
                [
                    shuffle(data["l_all"], random_state=i + random_state)
                    for i in range(n_episodes)
                ]
            )
        )
    elif training_schedule == "blocked":
        data["x_train"] = np.vstack(
            (
                np.vstack(
                    tuple(
                        [
                            shuffle(data["x_task_a"], random_state=i + random_state)
                            for i in range(n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["x_task_b"], random_state=i + random_state)
                            for i in range(n_episodes)
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
                            for i in range(n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["y_task_b"], random_state=i + random_state)
                            for i in range(n_episodes)
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
                            for i in range(n_episodes)
                        ]
                    )
                ),
                np.vstack(
                    tuple(
                        [
                            shuffle(data["l_task_b"], random_state=i + random_state)
                            for i in range(n_episodes)
                        ]
                    )
                ),
            )
        )
    else:
        print("Unknown training schedule parameter")

    if ctx_avg and ctx_avg_window > 0:
        data["x_train"][:, -2] = (
            pd.Series(data["x_train"][:, -2])
            .rolling(window=ctx_avg_window, min_periods=1)
            .mean()
        )
        data["x_train"][:, -1] = (
            pd.Series(data["x_train"][:, -1])
            .rolling(window=ctx_avg_window, min_periods=1)
            .mean()
        )

    if centering == True:
        sc = StandardScaler(with_std=False)
        data["x_train"] = sc.fit_transform(data["x_train"])
        data["x_task_a"] = sc.transform(data["x_task_a"])
        data["x_task_b"] = sc.transform(data["x_task_b"])
        data["x_all"] = sc.transform(data["x_all"])

    return data
