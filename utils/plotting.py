from copy import deepcopy
import pickle
from typing import Tuple, Union, List
import numpy as np
import matplotlib
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import ttest_ind
from utils import choicemodel
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS
from utils import eval
from hebbcl.parameters import parser
from utils import data
from sklearn.decomposition import PCA


def sem(x: np.array, ax: int) -> Union[float, np.array]:
    """calculates the standard error of the mean

    Args:
        x (np.array): dataset
        ax (int): axis along which to calculate the SEM

    Returns:
        Union[float, np.array]: SEM, can be array if input was matrix
    """

    return np.nanstd(x, ddof=1, axis=ax) / np.sqrt(np.shape(x)[ax])


def helper_ttest(p_blocked: np.array, p_interleaved: np.array, param: str) -> str:
    """conducts t-test, prints results to stdout and returns sigstars

    Args:
        p_blocked (np.array): some data
        p_interleaved (np.array): some other data
        param (str): name of data

    Returns:
        str: string with sigstars
    """
    res = ttest_ind(p_blocked, p_interleaved)
    z = res.statistic
    print(f"{param} blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    return sigstar


def disp_model_estimates(thetas: dict, cols: list = [[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]):
    """displays parameter estimates

    Args:
        thetas (dict): parameter estimates
        cols (list, optional): colours for barplots. Defaults to [[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]].
    """
    parameters = ["bias", "lapse", "slope", "offset"]

    plt.figure(figsize=(4, 1.5), dpi=300)
    plt.rcParams.update({"font.size": 6})

    for ii, param in enumerate(parameters):

        p_blocked = thetas["blocked"][param]
        p_interleaved = thetas["interleaved"][param]

        plt.subplot(1, 4, ii + 1)
        ax = plt.gca()

        ax.bar(0, p_blocked.mean(), yerr=sem(p_blocked, 0), color=cols[0], zorder=1)
        ax.bar(
            1, p_interleaved.mean(), yerr=sem(p_interleaved, 0), color=cols[1], zorder=1
        )

        ax.set(
            xticks=[0, 1],
            title=param,
        )
        ax.set_xticklabels(("blocked", "interleaved"), rotation=90)
        if param == "bias":
            ax.set_ylabel("angular bias (°)")
            ax.set_ylim((0, 20))
            sigstar = helper_ttest(p_blocked, p_interleaved, param)

            plt.plot([0, 1], [20, 20], "k-", linewidth=1)
            plt.text(0.5, 20, sigstar, ha="center", fontsize=6)
        elif param == "lapse":
            ax.set_ylabel("lapses (%)")
            ax.set_ylim((0, 0.51))
            ax.set_yticks([0, 0.25, 0.5])
            ax.set_yticklabels(np.round(ax.get_yticks() * 100, 2))
            sigstar = helper_ttest(p_blocked, p_interleaved, param)

            plt.plot([0, 1], [0.5, 0.5], "k-", linewidth=1)
            plt.text(0.5, 0.5, sigstar, ha="center", fontsize=6)
        elif param == "slope":
            ax.set_ylabel("slope (a.u)")
            ax.set_ylim((0, 15))
            sigstar = helper_ttest(p_blocked, p_interleaved, param)

            plt.plot([0, 1], [15, 15], "k-", linewidth=1)
            plt.text(0.5, 15, sigstar, ha="center", fontsize=6)
        elif param == "offset":
            ax.set_ylabel("offset (a.u.)")
            ax.set_ylim((-0.05, 0.2))
            sigstar = helper_ttest(p_blocked, p_interleaved, param)

            plt.plot([0, 1], [0.2, 0.2], "k-", linewidth=1)
            plt.text(0.5, 0.2, sigstar, ha="center", fontsize=6)
    sns.despine()

    plt.rc("figure", titlesize=6)
    plt.tight_layout()


def helper_make_colormap(
    basecols: np.array = np.array(
        [[63, 39, 24], [64, 82, 21], [65, 125, 18], [66, 168, 15], [68, 255, 10]]
    )
    / 255,
    n_items: int = 5,
    monitor: bool = False,
) -> Tuple[LinearSegmentedColormap, np.array]:
    """creates a colormap with custom values and spacing

    Args:
        basecols (np.array, optional): colours used for interpolation. Defaults to
        np.array([[63, 39, 24], [64, 82, 21], [65, 125, 18], [66, 168, 15], [68, 255, 10]])/255.
        n_items (int, optional): number of samples from cmap (i.e. resolution). Defaults to 5.
        monitor (bool, optional): plot results y/n. Defaults to False.

    Returns:
        Tuple[LinearSegmentedColormap, np.array]: segmented cmap (from matplotlib.colors), as well as
        array with rgb vals
    """

    # turn basecols into list of tuples
    basecols = list(map(lambda x: tuple(x), basecols))
    # turn basecols into colour map
    cmap = LinearSegmentedColormap.from_list("tmp", basecols, N=n_items)
    # if desired, plot results
    if monitor:
        plt.figure()
        plt.imshow(np.random.randn(20, 20), cmap=cmap)
        plt.colorbar()
    cols = np.asarray([list(cmap(c)[:3]) for c in range(n_items)])

    return cmap, cols


def plot_grid2(
    xy: np.array,
    line_colour: str = "green",
    line_width: int = 1,
    fig_id: int = 1,
    n_items: int = 5,
):

    # %matplotlib qt
    x, y = np.meshgrid(np.arange(0, n_items), np.arange(0, n_items))
    x = x.flatten()
    y = y.flatten()
    try:
        xy
    except NameError:
        xy = np.stack((x, y), axis=1)
    bl = np.stack((x, y), axis=1)
    plt.figure(fig_id)

    for ii in range(0, n_items - 1):
        for jj in range(0, n_items - 1):
            p1 = xy[(bl[:, 0] == ii) & (bl[:, 1] == jj), :].ravel()
            p2 = xy[(bl[:, 0] == ii + 1) & (bl[:, 1] == jj), :].ravel()
            plt.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], linewidth=line_width, color=line_colour
            )
            p2 = xy[(bl[:, 0] == ii) & (bl[:, 1] == jj + 1), :].ravel()
            plt.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], linewidth=line_width, color=line_colour
            )
    ii = n_items - 1
    for jj in range(0, n_items - 1):
        p1 = xy[(bl[:, 0] == ii) & (bl[:, 1] == jj), :].ravel()
        p2 = xy[(bl[:, 0] == ii) & (bl[:, 1] == jj + 1), :].ravel()
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], linewidth=line_width, color=line_colour
        )

    jj = n_items - 1
    for ii in range(0, n_items - 1):
        p1 = xy[(bl[:, 0] == ii) & (bl[:, 1] == jj), :].ravel()
        p2 = xy[(bl[:, 0] == ii + 1) & (bl[:, 1] == jj), :].ravel()
        plt.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], linewidth=line_width, color=line_colour
        )
    ax = plt.gca()
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def scatter_mds_2(
    xyz: np.array,
    task_id: str = "a",
    fig_id: int = 1,
    flipdims: bool = False,
    items_per_dim: int = 5,
    flipcols: bool = False,
    marker_scale: int = 1,
):

    if flipcols is True:
        col1 = (0, 0, 0.5)
        col2 = (255 / 255, 102 / 255, 0)
    else:
        col1 = (255 / 255, 102 / 255, 0)
        col2 = (0, 0, 0.5)

    if task_id == "both":
        n_items = items_per_dim ** 2 * 2
        ctxMarkerEdgeCol = [col1, col2]
    elif task_id == "a":
        n_items = items_per_dim ** 2
        ctxMarkerEdgeCol = col1
    elif task_id == "b":
        n_items = items_per_dim ** 2
        ctxMarkerEdgeCol = col2
    elif task_id == "avg":
        n_items = items_per_dim ** 2
        ctxMarkerEdgeCol = "k"

    ctxMarkerCol = "white"
    ctxMarkerSize = 4 * marker_scale
    scat_b = np.linspace(0.5, 2.5, items_per_dim) * marker_scale

    _, scat_l = helper_make_colormap(
        basecols=np.array(
            [[63, 39, 24], [64, 82, 21], [65, 125, 18], [66, 168, 15], [68, 255, 10]]
        )
        / 255,
        n_items=items_per_dim,
        monitor=False,
    )

    b, l = np.meshgrid(  # noqa E741
        np.arange(0, items_per_dim), np.arange(0, items_per_dim)
    )
    if flipdims is True:
        l, b = np.meshgrid(np.arange(0, items_per_dim), np.arange(0, items_per_dim))

    b = b.flatten()
    l = l.flatten()  # noqa E741
    x = xyz[:, 0]
    y = xyz[:, 1]

    if task_id == "both":
        b = np.concatenate((b, b), axis=0)
        l = np.concatenate((l, l), axis=0)  # noqa E741
        plt.figure(fig_id)

        for ii in range(0, n_items // 2):
            plt.plot(
                [x[ii], x[ii]],
                [y[ii], y[ii]],
                marker="s",
                markerfacecolor=ctxMarkerCol,
                markeredgecolor=ctxMarkerEdgeCol[0],
                markersize=ctxMarkerSize,
                markeredgewidth=0.5,
            )
            plt.plot(
                x[ii],
                y[ii],
                marker="h",
                markeredgecolor=scat_l[l[ii], :],
                markerfacecolor=scat_l[l[ii], :],
                markersize=scat_b[b[ii]],
            )

        for ii in range(n_items // 2, n_items):
            plt.plot(
                x[ii],
                y[ii],
                marker="s",
                markerfacecolor=ctxMarkerCol,
                markeredgecolor=ctxMarkerEdgeCol[1],
                markersize=ctxMarkerSize,
                markeredgewidth=0.5,
            )
            plt.plot(
                x[ii],
                y[ii],
                marker="h",
                markeredgecolor=scat_l[l[ii], :],
                markerfacecolor=scat_l[l[ii], :],
                markersize=scat_b[b[ii]],
            )
    else:
        for ii in range(0, n_items):

            plt.plot(
                x[ii],
                y[ii],
                marker="s",
                markerfacecolor=ctxMarkerCol,
                markeredgecolor=ctxMarkerEdgeCol,
                markersize=ctxMarkerSize,
                markeredgewidth=0.5,
            )
            plt.plot(
                x[ii],
                y[ii],
                marker="h",
                markeredgecolor=scat_l[l[ii], :],
                markerfacecolor=scat_l[l[ii], :],
                markersize=scat_b[b[ii]],
            )


def plot_MDS_embeddings_2D(
    embedding: np.array,
    fig: plt.figure,
    fig_id: int = 2,
    axlims: float = 2.5,
    flipdims: bool = False,
    monk: bool = False,
    flipcols: bool = False,
):

    if flipcols is True:
        col1 = (0, 0, 0.5)
        col2 = (255 / 255, 102 / 255, 0)
    else:
        col1 = (255 / 255, 102 / 255, 0)
        col2 = (0, 0, 0.5)

    if monk is True:
        n_items = 6
        ii_half = 36
    else:
        n_items = 5
        ii_half = 25

    plt.subplot(1, 2, 1)
    plot_grid2(
        embedding[0:ii_half, [0, 1]],
        line_width=0.5,
        line_colour=col1,
        fig_id=fig_id,
        n_items=n_items,
    )
    scatter_mds_2(
        embedding[0:ii_half, [0, 1]],
        fig_id=fig_id,
        task_id="a",
        flipdims=flipdims,
        items_per_dim=n_items,
        flipcols=flipcols,
    )
    plot_grid2(
        embedding[ii_half:, [0, 1]],
        line_width=0.5,
        line_colour=col2,
        fig_id=fig_id,
        n_items=n_items,
    )
    scatter_mds_2(
        embedding[ii_half:, [0, 1]],
        fig_id=fig_id,
        task_id="b",
        flipdims=flipdims,
        items_per_dim=n_items,
        flipcols=flipcols,
    )
    ax = plt.gca()
    ax.set_xlim([-axlims, axlims])
    ax.set_ylim([-axlims, axlims])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.xlabel("dim 1", fontsize=6)
    plt.ylabel("dim 2", fontsize=6)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_aspect("equal", "box")

    plt.subplot(1, 2, 2)
    plot_grid2(
        embedding[0:ii_half, [1, 2]],
        line_width=0.5,
        line_colour=col1,
        fig_id=fig_id,
        n_items=n_items,
    )
    scatter_mds_2(
        embedding[0:ii_half, [1, 2]],
        fig_id=fig_id,
        task_id="a",
        flipdims=flipdims,
        items_per_dim=n_items,
        flipcols=flipcols,
    )
    plot_grid2(
        embedding[ii_half:, [1, 2]],
        line_width=0.5,
        line_colour=col2,
        fig_id=fig_id,
        n_items=n_items,
    )
    scatter_mds_2(
        embedding[ii_half:, [1, 2]],
        fig_id=fig_id,
        task_id="b",
        flipdims=flipdims,
        items_per_dim=n_items,
        flipcols=flipcols,
    )
    ax = plt.gca()
    ax.set_xlim([-axlims, axlims])
    ax.set_ylim([-axlims, axlims])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    plt.xlabel("dim 2", fontsize=6)
    plt.ylabel("dim 3", fontsize=6)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_aspect("equal", "box")


def biplot(score, coeff, pcax, pcay, labels=None):
    """
    pyplot doesn't support biplots as matlab does. got this script from
        https://sukhbinder.wordpress.com/2015/08/05/biplot-with-python/

    USAGE: biplot(score,pca.components_,1,2,labels=categories)
    """
    pca1 = pcax - 1
    pca2 = pcay - 1
    xs = score[:, pca1]
    ys = score[:, pca2]
    n = score.shape[1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    print(n)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, pca1], coeff[i, pca2], color="r", alpha=0.5)
        if labels is None:
            plt.text(
                coeff[i, pca1] * 1.15,
                coeff[i, pca2] * 1.15,
                "Var" + str(i + 1),
                color="g",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, pca1] * 1.15,
                coeff[i, pca2] * 1.15,
                labels[i],
                color="g",
                ha="center",
                va="center",
            )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()


def plot_results(
    ws: np.array, delta_ws: np.array, n_trials: int, eta: float, sigma: float
):

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.plot([np.linalg.norm(wi) for wi in delta_ws])
    plt.title("norm of " + r"$\Delta w$")
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.subplot(2, 2, 2)
    plt.plot([np.linalg.norm(wi) for wi in ws])
    plt.title("norm of " + r"$w$")
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.subplot(2, 2, 3)
    _ = plt.plot([wi[-1] for wi in ws], color=[0, 0, 128 / 255])
    _ = plt.plot([wi[-2] for wi in ws], color=[255 / 255, 102 / 255, 0])
    plt.legend(["task a", "task b"])
    plt.title("context units")
    plt.xlabel("iter")
    plt.ylabel("weight")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.subplot(2, 2, 4)
    plt.imshow(np.asarray(ws).T)
    plt.xlabel("iter")
    plt.ylabel("input")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.title("all weights")
    plt.suptitle(
        r"$\eta = $" + f"{eta}" + r"  $\sigma_{init} = $" + f"{sigma}",
        fontweight="normal",
        fontsize=18,
    )
    plt.tight_layout()


def plot_initsign_results(
    ws: np.array,
    delta_ws: np.array,
    n_trials: int,
    eta: float,
    init_weights: list = [1, -1],
):

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot([np.linalg.norm(wi) for wi in delta_ws])
    plt.title("norm of " + r"$\Delta w$")
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.subplot(1, 3, 2)
    plt.plot([np.linalg.norm(wi) for wi in ws])
    plt.title("norm of " + r"$w$")
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.subplot(1, 3, 3)
    _ = plt.plot([wi[-1] for wi in ws], color=[0, 0, 128 / 255])
    _ = plt.plot([wi[-2] for wi in ws], color=[255 / 255, 102 / 255, 0])
    plt.legend(["task a", "task b"])
    plt.title("context weights")
    plt.xlabel("iter")
    plt.ylabel("weight")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")
    plt.suptitle(
        r"$\eta = $" + f"{eta}" + r"  $w_{init} = $" + f"{init_weights}",
        fontweight="normal",
        fontsize=16,
    )
    plt.tight_layout()


def plot_ghasla_results(
    data: dict,
    ws: np.array,
    delta_ws: np.array,
    n_trials: int,
    eta: float,
    sigma: float,
):

    # stats:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.subplot(2, 3, 1)
    plt.plot([np.linalg.norm(wi) for wi in delta_ws])
    plt.title("norm of " + r"$\Delta W$", fontsize=8)
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")

    plt.subplot(2, 3, 2)
    plt.plot([np.linalg.norm(wi) for wi in ws])
    plt.title("norm of " + r"$W$", fontsize=8)
    plt.xlabel("iter")
    plt.ylabel("norm")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")

    plt.subplot(2, 3, 3)
    a = plt.plot([wi[-2] for wi in ws], color=[0, 0, 128 / 255])
    b = plt.plot([wi[-1] for wi in ws], color=[255 / 255, 102 / 255, 0])
    plt.legend([a[0], b[0]], ["task a", "task b"], frameon=False)
    plt.title("context to hidden weights", fontsize=8)
    plt.xlabel("iter")
    plt.ylabel("weight")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")

    plt.subplot(2, 3, 4)
    plt.plot([np.corrcoef(wi[-1, :], wi[-2, :])[0, 1] for wi in ws])
    plt.title("correlation between ctx weight vectors", fontsize=8)
    plt.xlabel("iter")
    plt.ylabel("pearson's r")
    plt.plot([n_trials // 2, n_trials // 2], plt.ylim(), "k:")

    plt.subplot(2, 3, 5)
    ta = []
    tb = []
    for W in ws:
        ya = np.maximum((data["x_test_a"] @ W), 0).mean(0)
        yb = np.maximum((data["x_test_b"] @ W), 0).mean(0)
        ta.append(np.mean((ya > 0) & (yb == 0)))
        tb.append(np.mean((ya == 0) & (yb > 0)))
    plt.plot(ta, color=[0, 0, 128 / 255])
    plt.plot(tb, color=[255 / 255, 102 / 255, 0])
    plt.title("proportion of task-specific units", fontsize=8)
    plt.yticks(
        ticks=np.arange(0, 1.1, 0.1),
        labels=(np.arange(0, 1.1, 0.1) * 100).astype("int"),
    )
    plt.xlabel("iter")
    plt.ylabel("percentage")
    plt.legend(["task a", "task b"], frameon=False)
    plt.suptitle(
        "Learning Dynamics, "
        + r"$\eta = $"
        + f"{eta}"
        + r"  $\sigma_{init} = $"
        + f"{sigma}",
        fontweight="normal",
        fontsize=12,
    )
    plt.tight_layout()

    # connectivity matrix:
    plt.figure(figsize=(8, 3), dpi=300)
    plt.subplot(2, 1, 1)
    plt.imshow(ws[0])
    plt.ylabel("input unit")
    plt.xlabel("hidden unit / PC")
    plt.title("Initial Connectivity Matrix", fontsize=8)

    plt.subplot(2, 1, 2)
    plt.imshow(ws[-1])
    plt.ylabel("input unit")
    plt.xlabel("hidden unit / PC")
    plt.title("Endpoint Connectivity Matrix", fontsize=8)
    plt.suptitle("Weights", fontweight="normal", fontsize=12)

    plt.tight_layout()


def plot_basicstats(
    n_runs: int = 50,
    n_epochs: int = 200,
    models: list = ["baseline_interleaved_new_select", "baseline_blocked_new_select"],
    fixtreesbug: bool = False,
    
):
    """plots learning curves (acc, task selectivity, context corr) and choice mats

    Args:
        n_runs (int, optional): _description_. Defaults to 50.
        n_epochs (int, optional): _description_. Defaults to 200.
        models (list, optional): _description_. Defaults to ['baseline_interleaved_new_select',
        'baseline_blocked_new_select'].
    """
    if len(models) == 2:
        # acc
        f1, axs1 = plt.subplots(2, 1, figsize=(2.7, 3), dpi=300)
        # # unit alloc
        f2, axs2 = plt.subplots(2, 1, figsize=(2.7, 3), dpi=300)
        # # context corr
        f3, axs3 = plt.subplots(2, 1, figsize=(2.7, 3), dpi=300)
        # # choice matrices
        f4, axs4 = plt.subplots(2, 2, figsize=(5, 5), dpi=300)

        for i, m in enumerate(models):
            t_a = np.empty((n_runs, n_epochs))
            t_b = np.empty((n_runs, n_epochs))
            t_d = np.empty((n_runs, n_epochs))
            t_mixed = np.empty((n_runs, n_epochs))
            acc_1st = np.empty((n_runs, n_epochs))
            acc_2nd = np.empty((n_runs, n_epochs))
            contextcorr = np.empty((n_runs, n_epochs))
            cmats_a = []
            cmats_b = []

            for r in range(n_runs):
                with open(
                    "../checkpoints/" + m + "/run_" + str(r) + "/results.pkl", "rb"
                ) as f:
                    results = pickle.load(f)

                    # accuracy:
                    acc_1st[r, :] = results["acc_1st"]
                    acc_2nd[r, :] = results["acc_2nd"]
                    # task factorisation:
                    t_a[r, :] = results["n_only_b_regr"] / 100
                    t_b[r, :] = results["n_only_a_regr"] / 100
                    t_d[r, :] = results["n_dead"] / 100
                    t_mixed[r, :] = 1 - t_a[r, :] - t_b[r, :] - t_d[r, :]
                    # context correlation:
                    contextcorr[r, :] = results["w_context_corr"]
                    # outputs:

                    if fixtreesbug:
                        cc = np.clip(results["all_y_out"][1][:,], -709.78, 709.78).astype(
                            np.float64
                        )
                    else:
                        cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                            np.float64
                        )
                    choices = 1 / (1 + np.exp(-cc))
                    cmats_a.append(choices[:25].reshape(5, 5))
                    cmats_b.append(choices[25:].reshape(5, 5))

            cmats_a = np.array(cmats_a)
            cmats_b = np.array(cmats_b)

            # accuracy
            axs1[i].plot(
                np.arange(n_epochs), acc_1st.mean(0), color=[255 / 255, 102 / 255, 0]
            )
            axs1[i].fill_between(
                np.arange(n_epochs),
                acc_1st.mean(0) - np.std(acc_1st, 0) / np.sqrt(n_runs),
                acc_1st.mean(0) + np.std(acc_1st, 0) / np.sqrt(n_runs),
                alpha=0.5,
                color=[255 / 255, 102 / 255, 0],
                edgecolor=None,
            )
            axs1[i].plot(np.arange(n_epochs), acc_2nd.mean(0), color=[0, 0, 128 / 255])
            axs1[i].fill_between(
                np.arange(n_epochs),
                acc_2nd.mean(0) - np.std(acc_2nd, 0) / np.sqrt(n_runs),
                acc_2nd.mean(0) + np.std(acc_2nd, 0) / np.sqrt(n_runs),
                alpha=0.5,
                color=[0, 0, 128 / 255],
                edgecolor=None,
            )
            axs1[i].set_ylim([0.4, 1.05])
            axs1[i].set(xlabel="trial", ylabel="accuracy")
            axs1[i].legend(["1st task", "2nd task"], frameon=False)
            if "interleaved" not in m:
                axs1[i].plot([n_epochs / 2, n_epochs / 2], [0, 1], "k--", alpha=0.5)
            axs1[i].set_title(m.split("_")[1])
            plt.gcf()
            sns.despine(f1)
            f1.tight_layout()

            # unit allocation (task factorisation)
            axs2[i].plot(
                np.arange(n_epochs), t_b.mean(0), color=[255 / 255, 102 / 255, 0]
            )
            axs2[i].fill_between(
                np.arange(n_epochs),
                t_b.mean(0) - np.std(t_b, 0) / np.sqrt(n_runs),
                t_b.mean(0) + np.std(t_b, 0) / np.sqrt(n_runs),
                alpha=0.5,
                color=[255 / 255, 102 / 255, 0],
                edgecolor=None,
            )
            axs2[i].plot(np.arange(n_epochs), t_a.mean(0), color=[0, 0, 128 / 255])
            axs2[i].fill_between(
                np.arange(n_epochs),
                t_a.mean(0) - np.std(t_a, 0) / np.sqrt(n_runs),
                t_a.mean(0) + np.std(t_a, 0) / np.sqrt(n_runs),
                alpha=0.5,
                color=[0, 0, 128 / 255],
                edgecolor=None,
            )
            axs2[i].set_yticks([0, 0.5, 1])
            ticks = axs2[i].get_yticks()  # plt.yticks()
            axs2[i].set_yticklabels((int(x) for x in ticks * 100))
            axs2[i].set(xlabel="trial", ylabel="task-sel (%)")
            axs2[i].legend(["1st task", "2nd task"], frameon=False)
            if "interleaved" not in m:
                axs2[i].plot([n_epochs / 2, n_epochs / 2], [0, 1], "k--", alpha=0.5)
            axs2[i].set_title(m.split("_")[1])
            plt.gcf()
            sns.despine(f2)
            axs2[i].set_ylim([0, 1.05])
            f2.tight_layout()

            # context corr
            axs3[i].plot(np.arange(n_epochs), contextcorr.mean(0), color="k")
            axs3[i].fill_between(
                np.arange(n_epochs),
                contextcorr.mean(0) - np.std(contextcorr, 0) / np.sqrt(n_runs),
                contextcorr.mean(0) + np.std(contextcorr, 0) / np.sqrt(n_runs),
                alpha=0.5,
                color="magenta",
                edgecolor=None,
            )

            axs3[i].set_ylim([-1.1, 1.05])
            axs3[i].set(xlabel="trial", ylabel=r"$w_{context}$ corr ")
            if "interleaved" not in m:
                axs3[i].plot([n_epochs / 2, n_epochs / 2], [-1, 1], "k--", alpha=0.5)
            axs3[i].set_title(m.split("_")[1])
            sns.despine(f3)
            f3.tight_layout()

            # choice matrices
            axs4[i, 0].imshow(cmats_a.mean(0))
            axs4[i, 0].set_title("1st task")
            axs4[i, 0].set(
                xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="irrel", ylabel="rel"
            )
            axs4[i, 1].imshow(cmats_b.mean(0))
            axs4[i, 1].set(
                xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="rel", ylabel="irrel"
            )
            axs4[i, 1].set_title("2nd task")
            # PCM = axs4[i, 1].get_children()[
            #     -2
            # ]  # get the mappable, the 1st and the 2nd are the x and y axes

            # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            # plt.colorbar(PCM, cax=cax)
    elif len(models) == 1 or type(models) == str:
        if type(models) == list:
            m = models[0]
        else:
            m = models
        # acc
        f1, axs1 = plt.subplots(1, 1, figsize=(2.7, 2), dpi=300)
        # # unit alloc
        f2, axs2 = plt.subplots(1, 1, figsize=(2.7, 2), dpi=300)
        # # context corr
        f3, axs3 = plt.subplots(1, 1, figsize=(2.7, 2), dpi=300)
        # # choice matrices
        f4, axs4 = plt.subplots(1, 2, figsize=(5, 5), dpi=300)
        # # hidden layer MDS, interleaved

        t_a = np.empty((n_runs, n_epochs))
        t_b = np.empty((n_runs, n_epochs))
        t_d = np.empty((n_runs, n_epochs))
        t_mixed = np.empty((n_runs, n_epochs))
        acc_1st = np.empty((n_runs, n_epochs))
        acc_2nd = np.empty((n_runs, n_epochs))
        contextcorr = np.empty((n_runs, n_epochs))
        cmats_a = []
        cmats_b = []

        for r in range(n_runs):
            with open(
                "../checkpoints/" + m + "/run_" + str(r) + "/results.pkl", "rb"
            ) as f:
                results = pickle.load(f)

                # accuracy:
                acc_1st[r, :] = results["acc_1st"]
                acc_2nd[r, :] = results["acc_2nd"]
                # task factorisation:
                t_a[r, :] = results["n_only_b_regr"] / 100
                t_b[r, :] = results["n_only_a_regr"] / 100
                t_d[r, :] = results["n_dead"] / 100
                t_mixed[r, :] = 1 - t_a[r, :] - t_b[r, :] - t_d[r, :]
                # context correlation:
                contextcorr[r, :] = results["w_context_corr"]
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        cmats_a = np.array(cmats_a)
        cmats_b = np.array(cmats_b)

        # accuracy
        axs1.plot(np.arange(n_epochs), acc_1st.mean(0), color=[255 / 255, 102 / 255, 0])
        axs1.fill_between(
            np.arange(n_epochs),
            acc_1st.mean(0) - np.std(acc_1st, 0) / np.sqrt(n_runs),
            acc_1st.mean(0) + np.std(acc_1st, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color=[255 / 255, 102 / 255, 0],
            edgecolor=None,
        )
        axs1.plot(np.arange(n_epochs), acc_2nd.mean(0), color=[0, 0, 128 / 255])
        axs1.fill_between(
            np.arange(n_epochs),
            acc_2nd.mean(0) - np.std(acc_2nd, 0) / np.sqrt(n_runs),
            acc_2nd.mean(0) + np.std(acc_2nd, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color=[0, 0, 128 / 255],
            edgecolor=None,
        )
        axs1.set_ylim([0.4, 1.05])
        axs1.set(xlabel="trial", ylabel="accuracy")
        axs1.legend(["1st task", "2nd task"], frameon=False)
        if "interleaved" not in m:
            axs1.plot([n_epochs / 2, n_epochs / 2], [0, 1], "k--", alpha=0.5)
        axs1.set_title(m.split("_")[1])
        plt.gcf()
        sns.despine(f1)
        f1.tight_layout()

        # unit allocation (task factorisation)
        axs2.plot(np.arange(n_epochs), t_b.mean(0), color=[255 / 255, 102 / 255, 0])
        axs2.fill_between(
            np.arange(n_epochs),
            t_b.mean(0) - np.std(t_b, 0) / np.sqrt(n_runs),
            t_b.mean(0) + np.std(t_b, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color=[255 / 255, 102 / 255, 0],
            edgecolor=None,
        )
        axs2.plot(np.arange(n_epochs), t_a.mean(0), color=[0, 0, 128 / 255])
        axs2.fill_between(
            np.arange(n_epochs),
            t_a.mean(0) - np.std(t_a, 0) / np.sqrt(n_runs),
            t_a.mean(0) + np.std(t_a, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color=[0, 0, 128 / 255],
            edgecolor=None,
        )
        axs2.set_yticks([0, 0.5, 1])
        ticks = axs2.get_yticks()  # plt.yticks()
        axs2.set_yticklabels((int(x) for x in ticks * 100))
        axs2.set(xlabel="trial", ylabel="task-sel (%)")
        axs2.legend(["1st task", "2nd task"], frameon=False)
        if "interleaved" not in m:
            axs2.plot([n_epochs / 2, n_epochs / 2], [0, 1], "k--", alpha=0.5)
        axs2.set_title(m.split("_")[1])
        plt.gcf()
        sns.despine(f2)
        axs2.set_ylim([0, 1.05])
        f2.tight_layout()

        # context corr
        axs3.plot(np.arange(n_epochs), contextcorr.mean(0), color="k")
        axs3.fill_between(
            np.arange(n_epochs),
            contextcorr.mean(0) - np.std(contextcorr, 0) / np.sqrt(n_runs),
            contextcorr.mean(0) + np.std(contextcorr, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color="grey",
            edgecolor=None,
        )

        axs3.set_ylim([-1.1, 1.05])
        axs3.set(xlabel="trial", ylabel=r"$w_{context}$ corr ")
        if "interleaved" not in m:
            axs3.plot([n_epochs / 2, n_epochs / 2], [-1, 1], "k--", alpha=0.5)
        axs3.set_title(m.split("_")[1])
        sns.despine(f3)
        f3.tight_layout()

        # choice matrices

        axs4[0].imshow(cmats_a.mean(0))
        axs4[0].set_title("1st task")
        axs4[0].set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="irrel", ylabel="rel")
        axs4[1].imshow(cmats_b.mean(0))
        axs4[1].set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="rel", ylabel="irrel")
        axs4[1].set_title("2nd task")
        axs4[0].set_xticks([])
        axs4[0].set_yticks([])
        axs4[1].set_xticks([])
        axs4[1].set_yticks([])
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    f1.tight_layout()
    f2.tight_layout()
    f3.tight_layout()
    f4.tight_layout()
    f4.set_facecolor("w")


def plot_sluggish_results(
    alphas_to_plot: list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1],
    sluggish_vals: list = np.round(np.linspace(0.05, 1, 20), 2),
    cols: matplotlib.cm = cm.plasma([0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 0.9]),
    n_runs: int = 50,
    filename: str = "sluggish_baseline_int_select_sv",
):

    idces = [np.where(alp == sluggish_vals)[0][0] for alp in alphas_to_plot]

    # Accuracy
    f, axs = plt.subplots(1, 1, figsize=(3, 2.56), dpi=300)

    accs = []
    for pli, ii, sv, col in zip(np.arange(len(idces)), idces, alphas_to_plot, cols):
        acc_a = []
        acc_b = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                acc_a.append(results["acc_1st"][-1])
                acc_b.append(results["acc_2nd"][-1])
        accs.append((np.array(acc_a) + np.array(acc_b)) / 2)
    accs = np.flipud(np.array(accs))
    for acc, pli, ii, sv, col in zip(
        accs, np.arange(len(idces)), idces, alphas_to_plot, cols
    ):
        plt.bar(pli, acc.mean(), yerr=np.std(acc) / np.sqrt(n_runs), color=col)
        sns.despine()

    plt.xlabel(r"sluggishness ($1-\alpha$)")
    plt.title("Accuracy")
    plt.xticks(
        ticks=np.arange(7),
        labels=np.array(1 - np.fliplr(np.asarray([alphas_to_plot]))[0]).round(2),
    )
    plt.yticks(
        np.arange(0.4, 1.1, 0.2),
        labels=[int(x) for x in (np.arange(0.4, 1.1, 0.2) * 100)],
    )
    plt.ylabel("accuracy (%)")
    plt.ylim(0.4, 1)
    plt.tight_layout()

    # Choices
    f, axs = plt.subplots(1, 1, figsize=(3.81, 2.91), dpi=300)
    all_choices_rel = []
    all_choices_irrel = []
    for ii, sv, col in zip(idces, alphas_to_plot, cols):
        cmats_a = []
        cmats_b = []
        # corrs_run_b = []
        # corrs_run_i = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        cmats_a = np.array(cmats_a)
        cmats_b = np.array(cmats_b)
        choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
        choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2
        all_choices_rel.append(choices_rel)
        all_choices_irrel.append(choices_irrel)
    all_choices_rel.reverse()
    all_choices_irrel.reverse()
    for choices_rel, choices_irrel, ii, sv, col in zip(
        all_choices_rel, all_choices_irrel, idces, alphas_to_plot, cols
    ):
        plt.errorbar(
            np.arange(5),
            choices_rel.mean(0),
            yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
            marker="o",
            color=col,
            linestyle="",
            markersize=4,
        )
        plt.errorbar(
            np.arange(5),
            choices_irrel.mean(0),
            yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
            marker="o",
            color=col,
            linestyle="",
            markersize=4,
        )

        plt.plot(
            np.linspace(0, 4, 100),
            eval.sigmoid(
                np.linspace(-2, 2, 100),
                *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
            ),
            color=col,
            linestyle="-",
        )
        plt.plot(
            np.linspace(0, 4, 100),
            eval.sigmoid(
                np.linspace(-2, 2, 100),
                *choicemodel.fit_sigmoid(
                    zscore(np.arange(-2, 3)), choices_irrel.mean(0)
                ),
            ),
            color=col,
            linestyle="--",
        )

        sns.despine()
    cmap = cm.plasma
    norm = colors.Normalize(vmin=0.1, vmax=1)

    plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), label=r"sluggishness ($1-\alpha$)"
    )
    plt.title("psychometrics")
    plt.xlabel("feature value")
    plt.yticks(ticks=[0, 0.5, 1])
    plt.xticks(ticks=np.arange(5), labels=[1, 2, 3, 4, 5])
    plt.ylabel("p(accept)")
    plt.tight_layout()

    # Sigmoids
    f, axs = plt.subplots(1, 1, figsize=(3.81, 2.91), dpi=300)

    all_choices_rel = []
    all_choices_irrel = []
    for ii, sv, col in zip(idces, alphas_to_plot, cols):
        cmats_a = []
        cmats_b = []
        # corrs_run_b = []
        # corrs_run_i = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        cmats_a = np.array(cmats_a)
        cmats_b = np.array(cmats_b)
        choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
        choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2
        all_choices_rel.append(choices_rel)
        all_choices_irrel.append(choices_irrel)
    all_choices_rel.reverse()
    all_choices_irrel.reverse()
    for choices_rel, choices_irrel, ii, sv, col in zip(
        all_choices_rel, all_choices_irrel, idces, alphas_to_plot, cols
    ):
        plt.errorbar(
            np.arange(5),
            choices_rel.mean(0),
            yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
            marker="o",
            color=col,
            linestyle="",
            markersize=4,
        )
        plt.errorbar(
            np.arange(5),
            choices_irrel.mean(0),
            yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
            marker="o",
            color=col,
            linestyle="",
            markersize=4,
        )

        plt.plot(
            np.linspace(0, 4, 100),
            eval.sigmoid(
                np.linspace(-2, 2, 100),
                *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
            ),
            color=col,
            linestyle="-",
        )
        plt.plot(
            np.linspace(0, 4, 100),
            eval.sigmoid(
                np.linspace(-2, 2, 100),
                *choicemodel.fit_sigmoid(
                    zscore(np.arange(-2, 3)), choices_irrel.mean(0)
                ),
            ),
            color=col,
            linestyle="--",
        )

        sns.despine()
    cmap = cm.plasma
    norm = colors.Normalize(vmin=0.1, vmax=1)

    plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), label=r"sluggishness ($1-\alpha$)"
    )
    plt.title("psychometrics")
    plt.xlabel("feature value")
    plt.yticks(ticks=[0, 0.5, 1])
    plt.xticks(ticks=np.arange(5), labels=[1, 2, 3, 4, 5])
    plt.ylabel("p(accept)")
    plt.tight_layout()

    # choice matrices
    alphas_to_plot = [0.05, 0.1, 0.2, 0.3, 0.4, 0.7, 1]
    idces = [np.where(alp == sluggish_vals)[0][0] for alp in alphas_to_plot]

    unit_cols = [
        [250 / 255, 147 / 255, 30 / 255],
        [0 / 255, 6 / 255, 189 / 255],
        [4 / 255, 162 / 255, 201 / 255],
    ]

    n_runs = 50
    f, axs = plt.subplots(2, 7, figsize=(10, 4))
    all_cmats_a = []
    all_cmats_b = []

    for ax1, ax2, ii, sv in zip(axs[0, :], axs[1, :], idces, alphas_to_plot):
        cmats_a = []
        cmats_b = []
        # corrs_run_b = []
        # corrs_run_i = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        cmats_a = np.array(cmats_a)
        cmats_b = np.array(cmats_b)
        all_cmats_a.append(cmats_a)
        all_cmats_b.append(cmats_b)
    all_cmats_a.reverse()
    all_cmats_b.reverse()
    alphas_to_plot.reverse()
    for cmats_a, cmats_b, ax1, ax2, ii, sv in zip(
        all_cmats_a, all_cmats_b, axs[0, :], axs[1, :], idces, alphas_to_plot
    ):
        ax1.imshow(np.flipud(cmats_a.mean(0)))
        ax1.set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="irrel", ylabel="rel")
        [i.set_linewidth(3) for i in ax1.spines.values()]
        [i.set_color(unit_cols[0]) for i in ax1.spines.values()]
        ax2.imshow(np.flipud(cmats_b.mean(0)))
        ax2.set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="rel", ylabel="irrel")
        [i.set_linewidth(3) for i in ax2.spines.values()]
        [i.set_color(unit_cols[1]) for i in ax2.spines.values()]
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
    plt.tight_layout()

    # task factorisation

    rdms, dmat, _ = eval.gen_behav_models()

    all_betas = []
    for ii, sv in zip(idces, alphas_to_plot):
        betas = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
            choices = 1 / (1 + np.exp(-cc))
            yrdm = squareform(pdist(choices))

            y = zscore(yrdm[np.tril_indices(50, k=-1)]).flatten()
            assert len(y) == 1225
            lr = LinearRegression()
            lr.fit(dmat, y)
            betas.append(lr.coef_)
        all_betas.append(betas)
    all_betas.reverse()
    all_betas = np.array(all_betas)
    all_betas.shape

    plt.figure(figsize=(3.8, 2.9), dpi=300)
    for i in range(len(all_betas)):
        ha = plt.bar(
            i - 0.2,
            np.mean(all_betas[i, :, 0], 0),
            yerr=np.std(all_betas[i, :, 0], 0) / np.sqrt(all_betas.shape[1]),
            width=0.4,
            color="darkblue",
        )
        hb = plt.bar(
            i + 0.2,
            np.mean(all_betas[i, :, 1], 0),
            yerr=np.std(all_betas[i, :, 1], 0) / np.sqrt(all_betas.shape[1]),
            width=0.4,
            color="darkgreen",
        )
        sns.despine()
    plt.legend([ha, hb], ("factorised model", "linear model"), frameon=False)
    plt.ylabel(r"$\beta$ coefficient (a.u.)")
    plt.xlabel(r"sluggishness ($1-\alpha$)")
    plt.title("Choice Factorisation")
    plt.xticks(
        ticks=np.arange(7),
        labels=np.array(1 - np.fliplr(np.asarray([alphas_to_plot]))[0]).round(2),
    )

    # congruency effect

    all_accs = []
    _, _, cmats = eval.gen_behav_models()
    cmat_a = cmats[0, 0, :, :]
    cmat_b = cmats[0, 1, :, :]

    for ii, sv in zip(idces, alphas_to_plot):
        accs = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
            choices = 1 / (1 + np.exp(-cc))
            choices = choices.reshape((2, 5, 5))
            choices_a = choices[0, :, :]
            choices_b = choices[1, :, :]
            acc, aci = eval.compute_congruency_acc(choices_a, cmat_a)
            acc_a = acc - aci
            acc, aci = eval.compute_congruency_acc(choices_b, cmat_b)
            acc_b = acc - aci
            accs.append((acc_a + acc_b) / 2)

        all_accs.append(accs)
    all_accs.reverse()
    all_accs = np.array(all_accs)

    plt.figure(figsize=(3, 2.56), dpi=300)
    for acc, pli, ii, sv, col in zip(
        all_accs, np.arange(len(idces)), idces, alphas_to_plot, cols
    ):
        plt.bar(pli, acc.mean(), yerr=np.std(acc) / np.sqrt(n_runs), color=col)
        sns.despine()

    plt.xlabel(r"sluggishness ($1-\alpha$)")
    plt.title("Congruency Effect")
    plt.xticks(
        ticks=np.arange(7),
        labels=np.array(1 - np.fliplr(np.asarray([alphas_to_plot]))[0]).round(2),
    )
    plt.yticks(
        np.arange(0, 0.55, 0.1),
        labels=[int(x) for x in (np.arange(0, 0.55, 0.1) * 100)],
    )
    plt.ylabel("acc congr.-incongr. (%)")
    # plt.ylim(0.4,1)
    plt.tight_layout()

    # unit selectivity

    all_n_local = []
    for ii, sv in zip(idces, alphas_to_plot):
        n_local = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + filename
                + str(ii)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
            n_local.append(results["n_only_a_regr"][-1] + results["n_only_b_regr"][-1])

        all_n_local.append(n_local)
    all_n_local.reverse()
    all_n_local = np.array(all_n_local)

    plt.figure(figsize=(3, 2.56), dpi=300)
    for n_loc, pli, ii, sv, col in zip(
        all_n_local, np.arange(len(idces)), idces, alphas_to_plot, cols
    ):
        plt.bar(pli, n_loc.mean(), yerr=np.std(n_loc) / np.sqrt(n_runs), color=col)
        sns.despine()

    plt.xlabel(r"sluggishness ($1-\alpha$)")
    plt.title("Task Selectivity")
    plt.xticks(
        ticks=np.arange(7),
        labels=np.array(1 - np.fliplr(np.asarray([alphas_to_plot]))[0]).round(2),
    )
    plt.ylabel("task-sel. units (%)")
    plt.tight_layout()


def plot_mds(
    filename_embedding: str = "mds_embedding_baseline_int_new",
    filename_runs: str = "baseline_interleaved_new",
    thetas: tuple = (40, 0, 10),
    layer: str = "all_y_hidden",
    n_runs: int = 50,
    resultsdir: str = "../results/",
    axlims: int = 5,
    fixtreesbug: bool = True,
):
    """visualises hidden layer activity with MDS projection into 3 dim

    Args:
        filename_embedding (str, optional): name of mds file. Defaults to "mds_embedding_baseline_int_new".
        filename_runs (str, optional): name of results file. Defaults tp "baseline_interleaved_new".
        thetas (tuple, optional): rotation of mds projection. Defaults to (40,0,10).
        layer (str, optional). which layer to plot. Defaults to all_y_hidden.
        n_runs (int, optional): number of training runs. Defaults to 50.
        resultsdir (str, optional): location of training runs. Defaults to "../results/".
        axlims (int, optional): axis limits. Defaults to 5.
    """

    # check whether mds results already exist:
    try:
        with open(resultsdir + filename_embedding + ".pkl", "rb") as f:
            xyz = pickle.load(f)
    except FileNotFoundError:
        rdms = np.empty((n_runs, 50, 50))
        for r in range(n_runs):
            with open(
                "../checkpoints/" + filename_runs + "/run_" + str(r) + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
            if fixtreesbug:
                rdms[r, :, :] = squareform(
                    pdist(results[layer][-1][:, :], metric="euclidean")
                )

            else:
                rdms[r, :, :] = squareform(
                    pdist(results[layer][-1, :, :], metric="euclidean")
                )

        embedding = MDS(
            n_components=3,
            n_init=10,
            max_iter=10000,
            metric=True,
            dissimilarity="precomputed",
        )
        xyz = embedding.fit_transform(np.mean(rdms, 0))

        with open(resultsdir + filename_embedding + ".pkl", "wb") as f:
            pickle.dump(xyz, f)

    xyz_rot = eval.rotate(xyz, thetas[0], axis="x")
    xyz_rot = eval.rotate(xyz_rot, thetas[1], axis="y")
    xyz_rot = eval.rotate(xyz_rot, thetas[2], axis="z")

    plt.close()
    mm = 1 / 25.4
    fig = plt.figure(
        2, figsize=(69 * mm, 33 * mm), dpi=300, facecolor="w", edgecolor="k"
    )

    plot_MDS_embeddings_2D(xyz_rot, fig, fig_id=2, axlims=axlims)


def biplot_dataset(ds: str = "blobs", ctx_scaling: int = 6):
    """biplot of PCA performed on training dataset

    Args:
        ds (str, optional): dataset, can be "blobs" or "trees". Defaults to "blobs".
        ctx_scaling (int, optional): context scaling. Defaults to 6
    """
    args = parser.parse_args(args=[])
    args.n_episodes = 2
    args.ctx_scaling = ctx_scaling
    args.centering = True
    args.ctx_avg = False
    # biplot
    n_components = 27
    pca = PCA(n_components=n_components)
    if ds == "blobs":
        dataset = data.make_blobs_dataset(args)
    elif ds == "trees":
        dataset = data.make_trees_dataset(args)

    pca.fit(dataset["x_train"])
    # loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    scores = dataset["x_train"] @ pca.components_.T
    labels = [""] * (n_components - 2) + ["context1"] + ["context2"]
    score = scores
    coeff = pca.components_.T
    pcax = 1
    pcay = 2

    plt.figure(figsize=(3, 3), dpi=300)

    pca1 = pcax - 1
    pca2 = pcay - 1
    xs = score[:, pca1]
    ys = score[:, pca2]
    n = score.shape[1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    print(n)
    for i in range(n):
        plt.arrow(
            0, 0, coeff[i, pca1], coeff[i, pca2], color="r", alpha=0.5, head_width=0.05
        )
        if labels is None:
            plt.text(
                coeff[i, pca1] * 1.15,
                coeff[i, pca2] * 1.15,
                "Var" + str(i + 1),
                color="g",
                ha="center",
                va="center",
            )
        else:
            plt.text(
                coeff[i, pca1] * 1.15,
                coeff[i, pca2] * 1.15,
                labels[i],
                color="g",
                ha="center",
                va="center",
            )
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks(np.arange(-1, 1.1, 0.5))
    plt.yticks(np.arange(-1, 1.1, 0.5))
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    # plt.grid()
    sns.despine()
    plt.title("Biplot")


def plot_oja(
    n_hidden: int = 1,
    eta: float = 2e-1,
    sigma: float = 1e-2,
    n_episodes: int = 4,
    ds: str = "blobs",
):

    args = parser.parse_args(args=[])
    args.centering = True
    args.ctx_avg = False
    args.n_episodes = n_episodes
    if n_hidden == 1:
        eta = 4e-2
        sigma = 1e-3
        args.n_episodes = 1
        args.ctx_scaling = 5
    elif n_hidden == 100:
        eta = 2e-1
        sigma = 1e-2
        n_hidden = 100
        args.n_episodes = 4
        args.ctx_scaling = 1

    # n_trials = n_episodes * 50
    if ds == "blobs":
        dataset = data.make_blobs_dataset(args)
    elif ds == "trees":
        args.n_episodes += 1
        dataset = data.make_trees_dataset(args)

    if n_hidden > 1:
        W = np.random.randn(2, n_hidden) * sigma
        delta_ws = []
        ws = []
        delta_ws.append(0)
        ws.append(deepcopy(W))
        X = dataset["x_train"][:, -2:]
        X[X[:, 0] > 0, 1] = 0

        for x in X:
            x_vec = np.tile(x[:, np.newaxis], n_hidden)
            assert x_vec.T.shape == (n_hidden, 2)

            y = W.T @ x

            dW = eta * y * (x_vec - y * W)
            W += dW
            delta_ws.append(dW)
            ws.append(W)
    else:
        w = np.random.randn(dataset["x_train"].shape[1]) * sigma
        delta_ws = []
        ws = []
        delta_ws.append(0)
        ws.append(deepcopy(w))
        X = dataset["x_train"]
        for x in X:
            y = w.T @ x
            dw = eta * y * (x - y * w.T)
            w += dw
            delta_ws.append(dw)
            ws.append(deepcopy(w))

    plt.figure(figsize=(3, 2.5), dpi=300)

    a = plt.plot([wi[-1] for wi in ws], color=[255 / 255, 102 / 255, 0], linewidth=1)
    b = plt.plot([wi[-2] for wi in ws], color=[0, 0, 128 / 255], linewidth=1)
    plt.legend([a[0], b[0]], ["1st task", "2nd task"], frameon=False)

    plt.title("Task weights, Oja", fontsize=8)
    plt.xlabel("iter")
    plt.ylabel("weight")

    plt.ylabel("weight value (a.u.)")
    ax = plt.gca()
    for loc in ["top", "right"]:
        ax.spines[loc].set_visible(False)
    # plt.xlim([0,10])
    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.1, 1))


def plot_modelcomparison_accuracy(
    baseline_models: List[str] = [
        "baseline_interleaved_new_select",
        "baseline_blocked_new_select",
    ],
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.1,
    slope_blocked: int = 14,
    slope_int: int = 14,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
):
    pass
    # load slugglish sla int , collect accuracies
    idx = np.where(sluggish_vals == sluggishness)[0][0]
    tempval_blocked = np.logspace(np.log(0.1), np.log(4), 20)[slope_blocked]
    tempval_interleaved = np.logspace(np.log(0.1), np.log(4), 20)[slope_int]
    acc_int_oja = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_interleaved)
            acc_int_oja.append(
                choicemodel.compute_sampled_accuracy(
                    choices[:25].reshape(5, 5), choices[25:].reshape(5, 5)
                )
            )
    acc_int_oja = np.array(acc_int_oja)

    acc_blocked_oja = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_blocked)
            acc_blocked_oja.append(
                choicemodel.compute_sampled_accuracy(
                    choices[:25].reshape(5, 5), choices[25:].reshape(5, 5)
                )
            )
    acc_blocked_oja = np.array(acc_blocked_oja)

    # load baseline models:
    n_runs = 50
    acc_a = []
    acc_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[0] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            # acc_a.append(results['acc_1st_noise'][-1][6])
            acc_a.append(results["acc_1st"][-1])
            # acc_b.append(results['acc_2nd_noise'][-1][6])
            acc_b.append(results["acc_2nd"][-1])
    acc_int_baseline = (np.array(acc_a) + np.array(acc_b)) / 2
    n_runs = 50
    acc_a = []
    acc_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            # acc_a.append(results['acc_1st_noise'][-1][6])
            acc_a.append(results["acc_1st"][-1])
            # acc_b.append(results['acc_2nd_noise'][-1][6])
            acc_b.append(results["acc_2nd"][-1])
    acc_blocked_baseline = (np.array(acc_a) + np.array(acc_b)) / 2

    plt.figure(figsize=(3.2, 2.0), dpi=300)
    # make figure
    plt.subplot(1, 3, 1)
    accs = loadmat("../datasets/accs_exp1a.mat")
    plt.bar(
        np.arange(2),
        [accs["acc_b200"].mean(), accs["acc_int"].mean()],
        yerr=[
            np.std(accs["acc_b200"]) / np.sqrt(len(accs["acc_b200"].T)),
            np.std(accs["acc_int"]) / np.sqrt(len(accs["acc_int"].T)),
        ],
        color=[[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylim(0.5, 1.05)
    plt.yticks(
        ticks=np.arange(0.4, 1.1, 0.2), labels=np.arange(40, 101, 20), fontsize=6
    )
    plt.ylabel("accuracy (%)", fontsize=6)

    sns.despine()

    plt.title("Humans", fontsize=6)
    # statistical inference:
    res = ttest_ind(accs["acc_b200"].ravel(), accs["acc_int"].ravel())
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"acc humans blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 2)
    plt.bar(
        np.arange(2),
        [acc_blocked_baseline.mean(), acc_int_baseline.mean()],
        yerr=[
            np.std(acc_blocked_baseline) / np.sqrt(n_runs),
            np.std(acc_int_baseline) / np.sqrt(n_runs),
        ],
        color=["darkred", "red"],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylim(0.5, 1.05)
    plt.yticks(
        ticks=np.arange(0.4, 1.1, 0.2), labels=np.arange(40, 101, 20), fontsize=6
    )
    plt.ylabel("accuracy (%)", fontsize=6)
    # plt.xlabel('group',fontsize=6)
    sns.despine()
    plt.title("Baseline", fontsize=6)
    # statistical inference:
    res = ttest_ind(acc_blocked_baseline, acc_int_baseline)
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"acc baseline blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 3)
    plt.bar(
        np.arange(2),
        [acc_blocked_oja.mean(), acc_int_oja.mean()],
        yerr=[
            np.std(acc_blocked_oja) / np.sqrt(n_runs),
            np.std(acc_int_oja) / np.sqrt(n_runs),
        ],
        color=[[20 / 255, 78 / 255, 102 / 255], [50 / 255, 133 / 255, 168 / 255]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylim(0.5, 1.05)
    plt.yticks(
        ticks=np.arange(0.4, 1.1, 0.2), labels=np.arange(40, 101, 20), fontsize=6
    )
    plt.ylabel("accuracy (%)", fontsize=6)
    # plt.xlabel('group',fontsize=6)
    sns.despine()
    plt.title("EMA+Hebb", fontsize=6)

    # statistical inference:
    res = ttest_ind(acc_blocked_oja, acc_int_oja)
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"acc oja blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)
    plt.suptitle("Accuracy", fontsize=6)

    plt.tight_layout()


def plot_modelcomparison_sigmoids(
    baseline_models: List[str] = [
        "baseline_interleaved_new_select",
        "baseline_blocked_new_select",
    ],
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.1,
    slope_blocked: int = 14,
    slope_int: int = 14,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
) -> dict:
    # sigmoids
    idx = np.where(sluggish_vals == sluggishness)[0][0]
    tempval_blocked = np.logspace(np.log(0.1), np.log(4), 20)[slope_blocked]
    tempval_interleaved = np.logspace(np.log(0.1), np.log(4), 20)[slope_int]

    plt.figure(figsize=(4.8, 2.3), dpi=300)

    betas = {
        "humans": {
            "blocked": {"rel": [], "irrel": []},
            "int": {"rel": [], "irrel": []},
        },
        "baseline": {
            "blocked": {"rel": [], "irrel": []},
            "int": {"rel": [], "irrel": []},
        },
        "ema": {"blocked": {"rel": [], "irrel": []}, "int": {"rel": [], "irrel": []}},
    }

    choices = loadmat("../datasets/choices_exp1a.mat")
    # fit betas to individual subjects
    for k, v in choices.items():
        if "choices" in k:
            k = k.split("_")
            bs = []
            for y in v:
                bs.append(
                    choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False)
                )
            if k[1] == "b200":
                betas["humans"]["blocked"][k[-1]] = np.asarray(bs)
            else:
                betas["humans"][k[1]][k[-1]] = np.asarray(bs)

    plt.subplot(1, 3, 1)
    choices = loadmat("../datasets/choices_exp1a.mat")
    pl_bl_rel = plt.errorbar(
        np.arange(5),
        choices["choices_b200_rel"].mean(0),
        yerr=np.std(choices["choices_b200_rel"].mean(0)) / np.sqrt(n_runs),
        marker="o",
        color=(0.2, 0.2, 0.2),
        markersize=4,
        linestyle="",
    )
    pl_bl_irrel = plt.errorbar(
        np.arange(5),
        choices["choices_b200_irrel"].mean(0),
        yerr=np.std(choices["choices_b200_irrel"].mean(0)) / np.sqrt(n_runs),
        marker="*",
        color=(0.2, 0.2, 0.2),
        linestyle="",
        markersize=4,
    )
    f_bl_rel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(
                zscore(np.arange(-2, 3)), choices["choices_b200_rel"].mean(0)
            ),
        ),
        color=(0.2, 0.2, 0.2),
        linestyle="-",
    )
    f_bl_irrel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(
                zscore(np.arange(-2, 3)), choices["choices_b200_irrel"].mean(0)
            ),
        ),
        color=(0.2, 0.2, 0.2),
        linestyle="--",
    )

    pl_in_rel = plt.errorbar(
        np.arange(5),
        choices["choices_int_rel"].mean(0),
        yerr=np.std(choices["choices_int_rel"].mean(0)) / np.sqrt(n_runs),
        marker="o",
        color=(0.5, 0.5, 0.5),
        markersize=4,
        linestyle="",
    )
    pl_in_irrel = plt.errorbar(
        np.arange(5),
        choices["choices_int_irrel"].mean(0),
        yerr=np.std(choices["choices_int_irrel"].mean(0)) / np.sqrt(n_runs),
        marker="*",
        color=(0.5, 0.5, 0.5),
        linestyle="",
        markersize=4,
    )
    f_bl_rel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(
                zscore(np.arange(-2, 3)), choices["choices_int_rel"].mean(0)
            ),
        ),
        color=(0.5, 0.5, 0.5),
        linestyle="-",
    )
    f_bl_irrel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(
                zscore(np.arange(-2, 3)), choices["choices_int_irrel"].mean(0)
            ),
        ),
        color=(0.5, 0.5, 0.5),
        linestyle="--",
    )

    sns.despine()
    plt.legend(
        [pl_bl_rel[0], pl_bl_irrel[0], pl_in_rel[0], pl_in_irrel[0]],
        ["blocked - rel", "blocked - irrel", "int - rel", "int - irrel"],
        frameon=False,
        fontsize=6,
        loc=2,
    )
    plt.title("Humans", fontsize=6)
    plt.xlabel("feature value", fontsize=6)
    plt.ylabel("p(accept)", fontsize=6)
    plt.xticks(ticks=[0, 2, 4], labels=[1, 3, 5], fontsize=6)
    plt.yticks(ticks=(0, 0.5, 1), fontsize=6)

    plt.subplot(1, 3, 2)
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[0] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            # choices = choice_sigmoid(cc,T=tempval)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_a = np.array(cmats_a)
    cmats_b = np.array(cmats_b)
    choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
    choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2
    bs = []
    for y in choices_rel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["baseline"]["int"]["rel"] = np.asarray(bs)
    bs = []
    for y in choices_irrel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["baseline"]["int"]["irrel"] = np.asarray(bs)

    pl_in_rel = plt.errorbar(
        np.arange(5),
        choices_rel.mean(0),
        yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
        marker="d",
        color="red",
        markersize=4,
        linestyle="",
    )
    pl_in_irrel = plt.errorbar(
        np.arange(5),
        choices_irrel.mean(0),
        yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
        marker="*",
        color="red",
        linestyle="",
        markersize=4,
    )
    f_bl_rel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
        ),
        color="red",
        linestyle="-",
    )
    f_bl_irrel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_irrel.mean(0)),
        ),
        color="red",
        linestyle="--",
    )
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            # choices = choice_sigmoid(cc,T=tempval)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_a = np.array(cmats_a)
    cmats_b = np.array(cmats_b)
    choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
    choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2
    bs = []
    for y in choices_rel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["baseline"]["blocked"]["rel"] = np.asarray(bs)
    bs = []
    for y in choices_irrel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["baseline"]["blocked"]["irrel"] = np.asarray(bs)

    pl_bl_rel = plt.errorbar(
        np.arange(5),
        choices_rel.mean(0),
        yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
        marker="d",
        color="darkred",
        markersize=4,
        linestyle="",
    )
    pl_bl_irrel = plt.errorbar(
        np.arange(5),
        choices_irrel.mean(0),
        yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
        marker="*",
        color="darkred",
        linestyle="",
        markersize=4,
    )
    f_bl_rel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
        ),
        color="darkred",
        linestyle="-",
    )
    f_bl_irrel = plt.plot(
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_irrel.mean(0)),
        ),
        color="darkred",
        linestyle="--",
    )
    plt.legend(
        [pl_bl_rel[0], pl_bl_irrel[0], pl_in_rel[0], pl_in_irrel[0]],
        ["blocked - rel", "blocked - irrel", "int - rel", "int - irrel"],
        frameon=False,
        fontsize=6,
        loc=2,
    )
    plt.title("Baseline", fontsize=6)
    plt.xlabel("feature value", fontsize=6)
    plt.ylabel("p(accept)", fontsize=6)
    plt.xticks(ticks=[0, 2, 4], labels=[1, 3, 5], fontsize=6)
    plt.yticks(ticks=(0, 0.5, 1), fontsize=6)

    plt.subplot(1, 3, 3)
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_interleaved)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_a = np.array(cmats_a)
    cmats_b = np.array(cmats_b)
    choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
    choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2

    for y in choices_rel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["ema"]["int"]["rel"] = np.asarray(bs)
    bs = []
    for y in choices_irrel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["ema"]["int"]["irrel"] = np.asarray(bs)

    pl_in_rel = plt.errorbar(
        np.arange(5),
        choices_rel.mean(0),
        yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
        marker="o",
        color=[50 / 255, 133 / 255, 168 / 255],
        markersize=4,
        linestyle="",
    )
    pl_in_irrel = plt.errorbar(
        np.arange(5),
        choices_irrel.mean(0),
        yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
        marker="*",
        color=[50 / 255, 133 / 255, 168 / 255],
        linestyle="",
        markersize=4,
    )
    f_in_rel = plt.plot(  # noqa F841
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
        ),
        color=[50 / 255, 133 / 255, 168 / 255],
        linestyle="-",
    )
    f_in_irrel = plt.plot(  # noqa F841
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_irrel.mean(0)),
        ),
        color=[50 / 255, 133 / 255, 168 / 255],
        linestyle="--",
    )
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_blocked)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_a = np.array(cmats_a)
    cmats_b = np.array(cmats_b)
    choices_rel = (cmats_a.mean(2) + cmats_b.mean(1)) / 2
    choices_irrel = (cmats_a.mean(1) + cmats_b.mean(2)) / 2
    for y in choices_rel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["ema"]["blocked"]["rel"] = np.asarray(bs)
    bs = []
    for y in choices_irrel:
        bs.append(choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), y, fitlapse=False))
    betas["ema"]["blocked"]["irrel"] = np.asarray(bs)
    pl_bl_rel = plt.errorbar(
        np.arange(5),
        choices_rel.mean(0),
        yerr=np.std(choices_rel.mean(0)) / np.sqrt(n_runs),
        marker="o",
        color=[20 / 255, 78 / 255, 102 / 255],
        markersize=4,
        linestyle="",
    )
    pl_bl_irrel = plt.errorbar(
        np.arange(5),
        choices_irrel.mean(0),
        yerr=np.std(choices_irrel.mean(0)) / np.sqrt(n_runs),
        marker="*",
        color=[20 / 255, 78 / 255, 102 / 255],
        linestyle="",
        markersize=4,
    )
    f_bl_rel = plt.plot(  # noqa F841
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_rel.mean(0)),
        ),
        color=[20 / 255, 78 / 255, 102 / 255],
        linestyle="-",
    )
    f_bl_irrel = plt.plot(  # noqa F841
        np.linspace(0, 4, 100),
        eval.sigmoid(
            np.linspace(-2, 2, 100),
            *choicemodel.fit_sigmoid(zscore(np.arange(-2, 3)), choices_irrel.mean(0)),
        ),
        color=[20 / 255, 78 / 255, 102 / 255],
        linestyle="--",
    )

    sns.despine()
    plt.legend(
        [pl_bl_rel[0], pl_bl_irrel[0], pl_in_rel[0], pl_in_irrel[0]],
        ["blocked - rel", "blocked - irrel", "int - rel", "int - irrel"],
        frameon=False,
        fontsize=6,
        loc=2,
    )
    plt.title("EMA+Hebb", fontsize=6)
    plt.xlabel("feature value", fontsize=6)
    plt.ylabel("p(accept)", fontsize=6)
    plt.xticks(ticks=[0, 2, 4], labels=[1, 3, 5], fontsize=6)
    plt.yticks(ticks=(0, 0.5, 1), fontsize=6)
    plt.suptitle("Choices", fontsize=6)
    plt.tight_layout()

    return betas


def plot_modelcomparison_betas(betas: dict):

    plt.figure(figsize=(3.2, 2.0), dpi=300)
    # make figure
    plt.subplot(1, 3, 1)

    plt.bar(
        np.arange(2),
        [
            betas["humans"]["blocked"]["rel"][:, 1].mean(),
            betas["humans"]["int"]["rel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["humans"]["blocked"]["rel"][:, 1])
            / np.sqrt(len(betas["humans"]["blocked"]["rel"][:, 1].T)),
            np.std(betas["humans"]["int"]["rel"][:, 1])
            / np.sqrt(len(betas["humans"]["int"]["rel"][:, 1].T)),
        ],
        color=[[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.title("Humans", fontsize=6)
    res = ttest_ind(
        betas["humans"]["blocked"]["rel"][:, 1], betas["humans"]["int"]["rel"][:, 1]
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(z)
    print(f"betas rel humans blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [6, 6], "k-", linewidth=1)
    plt.text(0.5, 6, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 2)
    plt.bar(
        np.arange(2),
        [
            betas["baseline"]["blocked"]["rel"][:, 1].mean(),
            betas["baseline"]["int"]["rel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["baseline"]["blocked"]["rel"][:, 1])
            / np.sqrt(len(betas["baseline"]["blocked"]["rel"][:, 1].T)),
            np.std(betas["baseline"]["int"]["rel"][:, 1])
            / np.sqrt(len(betas["baseline"]["int"]["rel"][:, 1].T)),
        ],
        color=["darkred", "red"],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.title("Baseline", fontsize=6)
    res = ttest_ind(
        betas["baseline"]["blocked"]["rel"][:, 1], betas["baseline"]["int"]["rel"][:, 1]
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"betas rel baseline blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [11, 11], "k-", linewidth=1)
    plt.text(0.5, 11, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 3)
    plt.bar(
        np.arange(2),
        [
            betas["ema"]["blocked"]["rel"][:, 1].mean(),
            betas["ema"]["int"]["rel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["ema"]["blocked"]["rel"][:, 1])
            / np.sqrt(len(betas["ema"]["blocked"]["rel"][:, 1].T)),
            np.std(betas["ema"]["int"]["rel"][:, 1])
            / np.sqrt(len(betas["ema"]["int"]["rel"][:, 1].T)),
        ],
        color=[[20 / 255, 78 / 255, 102 / 255], [50 / 255, 133 / 255, 168 / 255]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()

    plt.title("EMA+Hebb", fontsize=6)
    res = ttest_ind(
        betas["ema"]["blocked"]["rel"][:, 1], betas["ema"]["int"]["rel"][:, 1]
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"betas rel ema blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [3, 3], "k-", linewidth=1)
    plt.text(0.5, 3, sigstar, ha="center", fontsize=6)
    plt.suptitle("Slopes, relevant dimension", fontsize=6)
    plt.tight_layout()

    plt.suptitle("Slopes, relevant dimension", fontsize=6)
    plt.tight_layout()

    plt.figure(figsize=(3.2, 2.0), dpi=300)
    # make figure
    plt.subplot(1, 3, 1)

    plt.bar(
        np.arange(2),
        [
            betas["humans"]["blocked"]["irrel"][:, 1].mean(),
            betas["humans"]["int"]["irrel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["humans"]["blocked"]["irrel"][:, 1])
            / np.sqrt(len(betas["humans"]["blocked"]["irrel"][:, 1].T)),
            np.std(betas["humans"]["int"]["irrel"][:, 1])
            / np.sqrt(len(betas["humans"]["int"]["irrel"][:, 1].T)),
        ],
        color=[[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.title("Humans", fontsize=6)
    res = ttest_ind(
        betas["humans"]["blocked"]["irrel"][:, 1], betas["humans"]["int"]["irrel"][:, 1]
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"betas irrel humans blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    else:
        sigstar = "*" * 4
    plt.plot([0, 1], [0.4, 0.4], "k-", linewidth=1)
    plt.text(0.5, 0.4, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 2)
    plt.bar(
        np.arange(2),
        [
            betas["baseline"]["blocked"]["irrel"][:, 1].mean(),
            betas["baseline"]["int"]["irrel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["baseline"]["blocked"]["irrel"][:, 1])
            / np.sqrt(len(betas["baseline"]["blocked"]["irrel"][:, 1].T)),
            np.std(betas["baseline"]["int"]["irrel"][:, 1])
            / np.sqrt(len(betas["baseline"]["int"]["irrel"][:, 1].T)),
        ],
        color=["darkred", "red"],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.title("Baseline", fontsize=6)
    res = ttest_ind(
        betas["baseline"]["blocked"]["irrel"][:, 1],
        betas["baseline"]["int"]["irrel"][:, 1],
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"betas irrel baseline blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    else:
        sigstar = "*" * 4
    plt.plot([0, 1], [1.2, 1.2], "k-", linewidth=1)
    plt.text(0.5, 1.2, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 3)
    plt.bar(
        np.arange(2),
        [
            betas["ema"]["blocked"]["irrel"][:, 1].mean(),
            betas["ema"]["int"]["irrel"][:, 1].mean(),
        ],
        yerr=[
            np.std(betas["ema"]["blocked"]["irrel"][:, 1])
            / np.sqrt(len(betas["ema"]["blocked"]["irrel"][:, 1].T)),
            np.std(betas["ema"]["int"]["irrel"][:, 1])
            / np.sqrt(len(betas["ema"]["int"]["irrel"][:, 1].T)),
        ],
        color=[[20 / 255, 78 / 255, 102 / 255], [50 / 255, 133 / 255, 168 / 255]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylabel("slope (a.u.)", fontsize=6)
    plt.yticks(fontsize=6)
    sns.despine()
    plt.title("EMA+Hebb", fontsize=6)
    res = ttest_ind(
        betas["ema"]["blocked"]["irrel"][:, 1], betas["ema"]["int"]["irrel"][:, 1]
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"betas irrel ema blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    else:
        sigstar = "*" * 4
    plt.plot([0, 1], [0.4, 0.4], "k-", linewidth=1)
    plt.text(0.5, 0.4, sigstar, ha="center", fontsize=6)

    plt.suptitle("Slopes, irrelevant dimension", fontsize=6)
    plt.tight_layout()


def plot_modelcomparison_choicemats(
    baseline_models: List[str] = [
        "baseline_interleaved_new_select",
        "baseline_blocked_new_select",
    ],
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.1,
    slope_blocked: int = 14,
    slope_int: int = 14,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
) -> dict:
    # sigmoids
    idx = np.where(sluggish_vals == sluggishness)[0][0]
    tempval_blocked = np.logspace(np.log(0.1), np.log(4), 20)[slope_blocked]
    tempval_interleaved = np.logspace(np.log(0.1), np.log(4), 20)[slope_int]

    cmats = loadmat("../datasets/choicemats_exp1a.mat", matlab_compatible=True)
    ks = list(cmats.keys())
    for k in ks:
        if "cmat" not in k:
            cmats.pop(k)
    cmats_human = {
        "int_a": cmats["cmat_i_north"],
        "int_b": cmats["cmat_i_south"],
        "bloc_a": cmats["cmat_b_north"],
        "bloc_b": cmats["cmat_b_south"],
    }

    # baseline cmats
    cmats_baseline = {}
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[0] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_baseline["int_a"] = np.array(cmats_a)
    cmats_baseline["int_b"] = np.array(cmats_b)

    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + baseline_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_baseline["bloc_a"] = np.array(cmats_a)
    cmats_baseline["bloc_b"] = np.array(cmats_b)

    # ema+sla cmats
    cmats_emasla = {}
    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_interleaved)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_emasla["int_a"] = np.array(cmats_a)
    cmats_emasla["int_b"] = np.array(cmats_b)

    cmats_a = []
    cmats_b = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl", "rb"
        ) as f:
            results = pickle.load(f)
            cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(np.float64)
            choices = 1 / (1 + np.exp(-cc))
            choices = choicemodel.choice_sigmoid(cc, T=tempval_blocked)
            cmats_a.append(choices[:25].reshape(5, 5))
            cmats_b.append(choices[25:].reshape(5, 5))

    cmats_emasla["bloc_a"] = np.array(cmats_a)
    cmats_emasla["bloc_b"] = np.array(cmats_b)

    # task factorisation (behav)
    betas_human = eval.compute_behav_taskfact(cmats_human)
    betas_baseline = eval.compute_behav_taskfact(cmats_baseline)
    betas_emasla = eval.compute_behav_taskfact(cmats_emasla)

    f = plt.figure(figsize=(4.8, 2.3), dpi=300)
    plt.subplot(1, 3, 1)
    # blocked
    ha = plt.bar(
        0 - 0.2,
        np.mean(betas_human["bloc"][:, 0], 0),
        yerr=np.std(betas_human["bloc"][:, 0], 0)
        / np.sqrt(betas_human["bloc"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        0 + 0.2,
        np.mean(betas_human["bloc"][:, 1], 0),
        yerr=np.std(betas_human["bloc"][:, 1], 0)
        / np.sqrt(betas_human["bloc"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )
    # interleaved
    ha = plt.bar(
        1 - 0.2,
        np.mean(betas_human["int"][:, 0], 0),
        yerr=np.std(betas_human["int"][:, 0], 0)
        / np.sqrt(betas_human["int"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        1 + 0.2,
        np.mean(betas_human["int"][:, 1], 0),
        yerr=np.std(betas_human["int"][:, 1], 0)
        / np.sqrt(betas_human["int"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )
    # stats factorised model
    res = ttest_ind(betas_human["bloc"][:, 0], betas_human["int"][:, 0])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(
        f"humans factorised model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}"
    )
    # stats linear models
    res = ttest_ind(betas_human["bloc"][:, 1], betas_human["int"][:, 1])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"humans linear model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")

    # stats interaction (with bars)
    res = ttest_ind(
        betas_human["bloc"][:, 0] - betas_human["bloc"][:, 1],
        betas_human["int"][:, 0] - betas_human["int"][:, 1],
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"humans interaction blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)

    sns.despine()
    plt.legend(
        [ha, hb], ("factorised model", "linear model"), frameon=False, fontsize=6
    )
    plt.ylabel(r"$\beta$ coefficient (a.u.)", fontsize=6)
    # plt.xlabel(r'sluggishness ($1-\alpha$)')
    plt.title("Humans", fontsize=6)
    plt.xticks([0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    ax = plt.gca()
    plt.yticks(
        ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=6
    )

    plt.subplot(1, 3, 2)
    # blocked
    ha = plt.bar(
        0 - 0.2,
        np.mean(betas_baseline["bloc"][:, 0], 0),
        yerr=np.std(betas_baseline["bloc"][:, 0], 0)
        / np.sqrt(betas_baseline["bloc"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        0 + 0.2,
        np.mean(betas_baseline["bloc"][:, 1], 0),
        yerr=np.std(betas_baseline["bloc"][:, 1], 0)
        / np.sqrt(betas_baseline["bloc"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )
    # interleaved
    ha = plt.bar(
        1 - 0.2,
        np.mean(betas_baseline["int"][:, 0], 0),
        yerr=np.std(betas_baseline["int"][:, 0], 0)
        / np.sqrt(betas_baseline["int"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        1 + 0.2,
        np.mean(betas_baseline["int"][:, 1], 0),
        yerr=np.std(betas_baseline["int"][:, 1], 0)
        / np.sqrt(betas_baseline["int"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )

    # stats factorised model
    res = ttest_ind(betas_baseline["bloc"][:, 0], betas_baseline["int"][:, 0])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(
        f"baseline factorised model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}"
    )
    # stats linear models
    res = ttest_ind(betas_baseline["bloc"][:, 1], betas_baseline["int"][:, 1])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(
        f"baseline linear model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}"
    )

    # stats interaction (with bars)
    res = ttest_ind(
        betas_baseline["bloc"][:, 0] - betas_baseline["bloc"][:, 1],
        betas_baseline["int"][:, 0] - betas_baseline["int"][:, 1],
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"baseline interaction blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)

    sns.despine()
    plt.legend(
        [ha, hb], ("factorised model", "linear model"), frameon=False, fontsize=6
    )
    plt.ylabel(r"$\beta$ coefficient (a.u.)", fontsize=6)
    # plt.xlabel(r'sluggishness ($1-\alpha$)')
    plt.title("Baseline", fontsize=6)
    plt.xticks([0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    ax = plt.gca()
    plt.yticks(
        ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=6
    )

    plt.subplot(1, 3, 3)
    # blocked
    ha = plt.bar(
        0 - 0.2,
        np.mean(betas_emasla["bloc"][:, 0], 0),
        yerr=np.std(betas_emasla["bloc"][:, 0], 0)
        / np.sqrt(betas_emasla["bloc"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        0 + 0.2,
        np.mean(betas_emasla["bloc"][:, 1], 0),
        yerr=np.std(betas_emasla["bloc"][:, 1], 0)
        / np.sqrt(betas_emasla["bloc"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )
    # interleaved
    ha = plt.bar(
        1 - 0.2,
        np.mean(betas_emasla["int"][:, 0], 0),
        yerr=np.std(betas_emasla["int"][:, 0], 0)
        / np.sqrt(betas_emasla["int"][:, 0].shape[0]),
        width=0.4,
        color="darkblue",
    )
    hb = plt.bar(
        1 + 0.2,
        np.mean(betas_emasla["int"][:, 1], 0),
        yerr=np.std(betas_emasla["int"][:, 1], 0)
        / np.sqrt(betas_emasla["int"][:, 1].shape[0]),
        width=0.4,
        color="darkgreen",
    )
    sns.despine()
    # stats factorised model
    res = ttest_ind(betas_emasla["bloc"][:, 0], betas_emasla["int"][:, 0])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"oja factorised model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    # stats linear models
    res = ttest_ind(betas_emasla["bloc"][:, 1], betas_emasla["int"][:, 1])
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"oja linear model blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")

    # stats interaction (with bars)
    res = ttest_ind(
        betas_emasla["bloc"][:, 0] - betas_emasla["bloc"][:, 1],
        betas_emasla["int"][:, 0] - betas_emasla["int"][:, 1],
    )
    z = res.statistic  # norm.isf(res.pvalue/2)
    print(f"oja interaction blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"

    plt.plot([0, 1], [1, 1], "k-", linewidth=1)
    plt.text(0.5, 1, sigstar, ha="center", fontsize=6)

    plt.legend(
        [ha, hb], ("factorised model", "linear model"), frameon=False, fontsize=6
    )
    plt.ylabel(r"$\beta$ coefficient (a.u.)", fontsize=6)
    # plt.xlabel(r'sluggishness ($1-\alpha$)')
    plt.title("EMA+Hebb", fontsize=6)
    plt.xticks([0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    ax = plt.gca()  # noqa F841
    plt.yticks(
        ticks=[0, 0.25, 0.5, 0.75, 1], labels=[0, 0.25, 0.5, 0.75, 1], fontsize=6
    )

    plt.tight_layout()
    return {"human": cmats_human, "baseline": cmats_baseline, "emahebb": cmats_emasla}


def plot_modelcomparison_choicemodel(
    baseline_models: List[str] = [
        "baseline_interleaved_new_select",
        "baseline_blocked_new_select",
    ],
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.1,
    slope_blocked: int = 14,
    slope_int: int = 14,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
):

    try:
        with open("../results/thetas_est.pkl", "rb") as f:
            all_thetas = pickle.load(f)
        disp_model_estimates(
            all_thetas["humans"], cols=([0.2, 0.2, 0.2], [0.5, 0.5, 0.5])
        )
        disp_model_estimates(all_thetas["baseline"], cols=["darkred", "red"])
        disp_model_estimates(
            all_thetas["emahebb"],
            cols=[[20 / 255, 78 / 255, 102 / 255], [50 / 255, 133 / 255, 168 / 255]],
        )
    except FileNotFoundError:

        # best fitting with oja slug models:
        idx = np.where(sluggish_vals == sluggishness)[0][0]
        tempidx_blocked = slope_blocked
        tempval_blocked = np.logspace(np.log(0.1), np.log(4), 20)[tempidx_blocked]
        tempidx_interleaved = slope_int
        tempval_interleaved = np.logspace(np.log(0.1), np.log(4), 20)[
            tempidx_interleaved
        ]

        cmats = loadmat("../datasets/choicemats_exp1a.mat")

        choicemats_humans = {
            "blocked": {
                "task_a": cmats["cmat_b_south"],
                "task_b": cmats["cmat_b_north"],
            },
            "interleaved": {
                "task_a": cmats["cmat_i_south"],
                "task_b": cmats["cmat_i_north"],
            },
        }

        choicemats_baseline = {
            "blocked": {"task_a": [], "task_b": []},
            "interleaved": {"task_a": [], "task_b": []},
        }

        choicemats_sla = {
            "blocked": {"task_a": [], "task_b": []},
            "interleaved": {"task_a": [], "task_b": []},
        }

        # baseline --------------------------------

        cmats_a = []
        cmats_b = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + baseline_models[1]
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        choicemats_baseline["blocked"]["task_b"] = np.array(cmats_a)
        choicemats_baseline["blocked"]["task_a"] = np.array(cmats_b)

        cmats_a = []
        cmats_b = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + baseline_models[0]
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        choicemats_baseline["interleaved"]["task_b"] = np.array(cmats_a)
        choicemats_baseline["interleaved"]["task_a"] = np.array(cmats_b)

        # oja ------------------------------------------

        cmats_a = []
        cmats_b = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                choices = choicemodel.choicemodel.choice_sigmoid(cc, T=tempval_blocked)
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        choicemats_sla["blocked"]["task_b"] = np.array(cmats_a)
        choicemats_sla["blocked"]["task_a"] = np.array(cmats_b)

        cmats_a = []
        cmats_b = []
        for r in np.arange(0, n_runs):
            with open(
                "../checkpoints/"
                + hebb_models[0]
                + str(idx)
                + "/run_"
                + str(r)
                + "/results.pkl",
                "rb",
            ) as f:
                results = pickle.load(f)
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                choices = choicemodel.choice_sigmoid(cc, T=tempval_interleaved)
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        choicemats_sla["interleaved"]["task_b"] = np.array(cmats_a)
        choicemats_sla["interleaved"]["task_a"] = np.array(cmats_b)

        thetas_humans = {}
        for cur in ["blocked", "interleaved"]:
            thetas_humans[cur] = choicemodel.fit_model_to_subjects(
                choicemats_humans[cur], n_runs=1
            )

        thetas_baseline = {}
        for cur in ["blocked", "interleaved"]:
            thetas_baseline[cur] = choicemodel.fit_model_to_subjects(
                choicemats_baseline[cur], n_runs=1
            )

        thetas_oja = {}
        for cur in ["blocked", "interleaved"]:
            thetas_oja[cur] = choicemodel.fit_model_to_subjects(
                choicemats_sla[cur], n_runs=1
            )

        all_thetas = {
            "humans": thetas_humans,
            "baseline": thetas_baseline,
            "emahebb": thetas_oja,
        }

        # all_thetas['emahebb'] = thetas_oja
        with open("../results/thetas_est.pkl", "wb") as f:
            pickle.dump(all_thetas, f)


def plot_modelcomparison_congruency(cmats: dict):
    _, _, cmats_model = eval.gen_behav_models()
    cmat_a = cmats_model[0, 0, :, :]
    cmat_b = cmats_model[0, 1, :, :]
    # human
    acc_congr_human = {}
    curricula = ("bloc", "int")
    for c in curricula:
        acc_congr_human[c] = []
        for ii in range(len(cmats["human"][c + "_a"])):
            c_a = cmats["human"][c + "_a"][ii, :, :]
            c_b = cmats["human"][c + "_b"][ii, :, :]
            # compute congruency effect:
            accc_a, acci_a = eval.compute_congruency_acc(c_a, cmat_a)
            accc_b, acci_b = eval.compute_congruency_acc(c_b, cmat_b)
            acc_congr_human[c].append([(accc_a + accc_b) / 2 - (acci_a + acci_b) / 2])
        acc_congr_human[c] = np.asarray(acc_congr_human[c])

    # baseline
    acc_congr_baseline = {}
    curricula = ("bloc", "int")
    for c in curricula:
        acc_congr_baseline[c] = []
        for ii in range(len(cmats["baseline"][c + "_a"])):
            c_a = cmats["baseline"][c + "_a"][ii, :, :]
            c_b = cmats["baseline"][c + "_b"][ii, :, :]
            # compute congruency effect:
            accc_a, acci_a = eval.compute_congruency_acc(c_a, cmat_a)
            accc_b, acci_b = eval.compute_congruency_acc(c_b, cmat_b)
            acc_congr_baseline[c].append(
                [(accc_a + accc_b) / 2 - (acci_a + acci_b) / 2]
            )
        acc_congr_baseline[c] = np.asarray(acc_congr_baseline[c])

    # ema+sla
    acc_congr_emasla = {}
    curricula = ("bloc", "int")
    for c in curricula:
        acc_congr_emasla[c] = []
        for ii in range(len(cmats["emahebb"][c + "_a"])):
            c_a = cmats["emahebb"][c + "_a"][ii, :, :]
            c_b = cmats["emahebb"][c + "_b"][ii, :, :]
            # compute congruency effect:
            accc_a, acci_a = eval.compute_congruency_acc(c_a, cmat_a)
            accc_b, acci_b = eval.compute_congruency_acc(c_b, cmat_b)
            acc_congr_emasla[c].append([(accc_a + accc_b) / 2 - (acci_a + acci_b) / 2])
        acc_congr_emasla[c] = np.asarray(acc_congr_emasla[c])

    plt.figure(figsize=(3.2, 2.0), dpi=300)
    # make figure
    plt.subplot(1, 3, 1)
    # accs = loadmat("../datasets/accs_exp1a.mat")
    plt.bar(
        np.arange(2),
        [acc_congr_human["bloc"].mean(), acc_congr_human["int"].mean()],
        yerr=[
            np.std(acc_congr_human["bloc"]) / np.sqrt(len(acc_congr_human["bloc"])),
            np.std(acc_congr_human["int"]) / np.sqrt(len(acc_congr_human["int"])),
        ],
        color=[[0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylim((0, 0.5))
    plt.yticks(
        ticks=plt.yticks()[0],
        labels=(int(x) for x in plt.yticks()[0] * 100),
        fontsize=6,
    )
    plt.ylabel("accuracy difference (%)", fontsize=6)
    sns.despine()
    plt.title("Humans", fontsize=6)
    res = ttest_ind(acc_congr_human["bloc"].ravel(), acc_congr_human["int"].ravel())
    z = res.statistic  # norm.isf(res.pvalue / 2)
    print(f"humans, congruency blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0, 1], [0.5, 0.5], "k-", linewidth=1)
    plt.text(0.5, 0.5, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 2)
    plt.bar(
        np.arange(2),
        [acc_congr_baseline["bloc"].mean(), acc_congr_baseline["int"].mean()],
        yerr=[
            np.std(acc_congr_baseline["bloc"])
            / np.sqrt(len(acc_congr_baseline["bloc"])),
            np.std(acc_congr_baseline["int"]) / np.sqrt(len(acc_congr_baseline["int"])),
        ],
        color=["darkred", "red"],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.yticks(
        ticks=plt.yticks()[0],
        labels=(int(x) for x in plt.yticks()[0] * 100),
        fontsize=6,
    )
    plt.ylabel("accuracy difference (%)", fontsize=6)
    plt.ylim((0, 0.5))
    sns.despine()
    plt.title("Baseline", fontsize=6)
    res = ttest_ind(
        acc_congr_baseline["bloc"].ravel(), acc_congr_baseline["int"].ravel()
    )
    z = res.statistic  # norm.isf(res.pvalue / 2)
    print(
        f"baselines, congruency blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}"
    )
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0, 1], [0.5, 0.5], "k-", linewidth=1)
    plt.text(0.5, 0.5, sigstar, ha="center", fontsize=6)

    plt.subplot(1, 3, 3)
    plt.bar(
        np.arange(2),
        [acc_congr_emasla["bloc"].mean(), acc_congr_emasla["int"].mean()],
        yerr=[
            np.std(acc_congr_emasla["bloc"]) / np.sqrt(len(acc_congr_emasla["bloc"])),
            np.std(acc_congr_emasla["int"]) / np.sqrt(len(acc_congr_emasla["int"])),
        ],
        color=[[20 / 255, 78 / 255, 102 / 255], [0.5 / 255, 133 / 255, 168 / 255]],
        width=0.8,
    )
    plt.xticks(ticks=[0, 1], labels=["blocked", "interleaved"], rotation=90, fontsize=6)
    plt.ylim((0, 0.5))
    plt.yticks(
        ticks=plt.yticks()[0],
        labels=(int(x) for x in plt.yticks()[0] * 100),
        fontsize=6,
    )
    plt.ylabel("accuracy difference (%)", fontsize=6)
    sns.despine()
    plt.title("EMA+Hebb", fontsize=6)
    res = ttest_ind(acc_congr_emasla["bloc"].ravel(), acc_congr_emasla["int"].ravel())
    z = res.statistic  # norm.isf(res.pvalue / 2)
    print(f"oja, congruency blocked vs interleaved: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0, 1], [0.5, 0.5], "k-", linewidth=1)
    plt.text(0.5, 0.5, sigstar, ha="center", fontsize=6)

    plt.suptitle("Congruency Effect", fontsize=6)
    plt.tight_layout()


def plot_modelcomparison_hiddenlayerRSA(
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.48,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
) -> dict:
    idx = np.where(sluggish_vals == sluggishness)[0][0]
    betas_blocked = []
    betas_int = []
    rdms, dmat = eval.gen_modelrdms(ctx_offset=6)

    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl", "rb"
        ) as f:
            results = pickle.load(f)
        y = zscore(
            squareform(pdist(results["all_y_hidden"][1, :, :]))[
                np.tril_indices(50, k=-1)
            ]
        ).flatten()
        lr = LinearRegression()
        lr.fit(dmat, y)
        betas_blocked.append(lr.coef_)
    betas_blocked = np.asarray(betas_blocked)

    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
        y = zscore(
            squareform(pdist(results["all_y_hidden"][1, :, :]))[
                np.tril_indices(50, k=-1)
            ]
        ).flatten()
        lr = LinearRegression()
        lr.fit(dmat, y)
        betas_int.append(lr.coef_)
    betas_int = np.asarray(betas_int)

    f, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=300)

    b1 = axs.bar(
        0 - 0.2,
        betas_blocked[:, 0].mean(0),
        yerr=np.std(betas_blocked[:, 0], 0) / np.sqrt(n_runs),
        color=(39 / 255, 140 / 255, 145 / 255),
        width=0.2,
    )
    b2 = axs.bar(
        0,
        betas_blocked[:, 1].mean(0),
        yerr=np.std(betas_blocked[:, 1], 0) / np.sqrt(n_runs),
        color=(34 / 255, 76 / 255, 128 / 255),
        width=0.2,
    )
    b3 = axs.bar(
        0 + 0.2,
        betas_blocked[:, 2].mean(0),
        yerr=np.std(betas_blocked[:, 2], 0) / np.sqrt(n_runs),
        color=(159 / 255, 45 / 255, 235 / 255),
        width=0.2,
    )

    b1 = axs.bar(
        1 - 0.2,
        betas_int[:, 0].mean(0),
        yerr=np.std(betas_int[:, 0], 0) / np.sqrt(n_runs),
        color=(39 / 255, 140 / 255, 145 / 255),
        width=0.2,
    )
    b2 = axs.bar(
        1,
        betas_int[:, 1].mean(0),
        yerr=np.std(betas_int[:, 1], 0) / np.sqrt(n_runs),
        color=(34 / 255, 76 / 255, 128 / 255),
        width=0.2,
    )
    b3 = axs.bar(
        1 + 0.2,
        betas_int[:, 2].mean(0),
        yerr=np.std(betas_int[:, 2], 0) / np.sqrt(n_runs),
        color=(159 / 255, 45 / 255, 235 / 255),
        width=0.2,
    )
    axs.set_xticks((0, 1))
    axs.set_yticks((0, 0.25, 0.5, 0.75, 1))
    axs.set_xticklabels(("blocked", "interleaved"), rotation=0, fontsize=6)
    axs.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=6)
    axs.set_title("Model RSA, Hidden Layer", fontsize=6)
    axs.set_ylabel(r"$\beta$ estimate (a.u.)", fontsize=6)
    axs.legend(
        [b1, b2, b3], ["grid", "orthogonal", "diagonal"], fontsize=6, frameon=False
    )

    res = ttest_ind(betas_blocked[:, 0].ravel(), betas_blocked[:, 1].ravel())
    z = res.statistic
    print(f"blocked grid vs orth: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([-0.2, 0], [0.75, 0.75], "k-", linewidth=1)
    plt.text(-0.1, 0.75, sigstar, ha="center", fontsize=6)

    res = ttest_ind(betas_blocked[:, 1].ravel(), betas_blocked[:, 2].ravel())
    z = res.statistic
    print(f"blocked orth vs diag: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0, 0.2], [0.7, 0.7], "k-", linewidth=1)
    plt.text(0.1, 0.7, sigstar, ha="center", fontsize=6)

    res = ttest_ind(betas_int[:, 0].ravel(), betas_int[:, 2].ravel())
    z = res.statistic
    print(f"int grid vs diag: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0.8, 1.2], [0.94, 0.94], "k-", linewidth=1)
    plt.text(1, 0.94, sigstar, ha="center", fontsize=6)

    res = ttest_ind(betas_int[:, 1].ravel(), betas_int[:, 2].ravel())
    z = res.statistic
    print(f"int orth vs diag: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([1, 1.2], [0.88, 0.88], "k-", linewidth=1)
    plt.text(1.1, 0.88, sigstar, ha="center", fontsize=6)

    sns.despine()
    plt.tight_layout()


def plot_modelcomparison_taskselectivity(
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.48,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
):
    idx = np.where(sluggish_vals == sluggishness)[0][0]
    blocked_n_only_a = []
    blocked_n_only_b = []
    blocked_n_mixed = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl", "rb"
        ) as f:
            results = pickle.load(f)
        blocked_n_only_a.append(results["n_only_a_regr"][-1])
        blocked_n_only_b.append(results["n_only_b_regr"][-1])
        blocked_n_mixed.append(100 - blocked_n_only_a[-1] - blocked_n_only_b[-1])

    blocked_n_only_a = np.asarray(blocked_n_only_a)
    blocked_n_only_b = np.asarray(blocked_n_only_b)
    blocked_n_mixed = np.asarray(blocked_n_mixed)
    print(f"blocked, task-specific: {blocked_n_only_a.mean()+blocked_n_only_b.mean()}")
    print(f"blocked, task-agnostic: {blocked_n_mixed.mean()}")
    int_n_only_a = []
    int_n_only_b = []
    int_n_mixed = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
        int_n_only_a.append(results["n_only_a_regr"][-1])
        int_n_only_b.append(results["n_only_b_regr"][-1])
        int_n_mixed.append(100 - int_n_only_a[-1] - int_n_only_b[-1])

    int_n_only_a = np.asarray(int_n_only_a)
    int_n_only_b = np.asarray(int_n_only_b)
    int_n_mixed = np.asarray(int_n_mixed)
    print(f"int, task-specific: {int_n_only_a.mean()+int_n_only_b.mean()}")
    print(f"int, task-agnostic: {int_n_mixed.mean()}")
    n_a = np.stack((blocked_n_only_a, int_n_only_a), axis=1)
    n_b = np.stack((blocked_n_only_b, int_n_only_b), axis=1)
    n_m = np.stack((blocked_n_mixed, int_n_mixed), axis=1)
    f, ax = plt.subplots(figsize=(2, 2), dpi=300)
    b1 = ax.bar(
        ["blocked", "interleaved"],
        n_a.mean(0),
        yerr=np.std(n_a, 0, ddof=1) / np.sqrt(n_runs),
        width=0.2,
    )
    b2 = ax.bar(
        ["blocked", "interleaved"],
        n_b.mean(0),
        yerr=np.std(n_a, 0, ddof=1) / np.sqrt(n_runs),
        bottom=n_a.mean(0),
        width=0.2,
    )
    b3 = ax.bar(
        ["blocked", "interleaved"],
        n_m.mean(0),
        yerr=np.std(n_a, 0, ddof=1) / np.sqrt(n_runs),
        bottom=n_b.mean(0) + n_a.mean(0),
        width=0.2,
    )
    ax.set_ylabel("hidden units (%)", fontsize=6)
    ax.legend(
        [b1, b2, b3],
        ["1st task, rel. dim.", "2nd task, rel. dim.", "task agnostic"],
        fontsize=6,
        frameon=False,
    )
    ax.set_title("Hidden Unit Selectivity", fontsize=6)
    sns.despine()

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + ax.get_legend().get_texts()
    ):
        item.set_fontsize(6)


def plot_modelcomparison_readout(
    ds: str = "blobs",
    ctx_scaling: int = 1,
    hebb_models: List[str] = [
        "sluggish_oja_int_select_sv",
        "oja_blocked_new_select_halfcenter",
    ],
    sluggishness: float = 0.48,
    n_runs: int = 20,
    sluggish_vals: np.array = np.round(np.linspace(0.05, 1, 20), 2),
):

    idx = np.where(sluggishness == sluggish_vals)[0][0]

    args = parser.parse_args(args=[])
    args.n_episodes = 1
    args.ctx_scaling = ctx_scaling
    args.centering = False
    args.ctx_avg = False
    if ds == "blobs":
        dataset = data.make_blobs_dataset(args)
    elif ds == "trees":
        dataset = data.make_trees_dataset(args)

    
    dmat = eval.make_dmat(dataset["f_all"])
    

    readout_magnitude = np.empty((50, 3))
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)

        yh = np.array(results["all_y_hidden"][1, :, :],dtype=np.float64)
        selectivity_matrix = np.zeros((100, 6))
        for i_neuron in range(100):
            y_neuron = yh[:, i_neuron]           
            lr = sm.OLS(zscore(y_neuron), dmat)
            regr_results = lr.fit()
            # if only a single regressor is significant, store that neuron's selectivity
            if np.sum(regr_results.tvalues > 1.96) == 1:
                selectivity_matrix[
                    i_neuron,
                    np.where(regr_results.tvalues == np.max(regr_results.tvalues))[0][
                        0
                    ],
                ] = 1
        i_task_a = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 1)
            & (selectivity_matrix[:, 2] == 0)
            & (selectivity_matrix[:, 3] == 0)
        )
        i_task_b = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 0)
            & (selectivity_matrix[:, 2] == 1)
            & (selectivity_matrix[:, 3] == 0)
        )

        with open(
            "../checkpoints/"
            + hebb_models[0]
            + str(idx)
            + "/run_"
            + str(r)
            + "/model.pkl",
            "rb",
        ) as f:
            model = pickle.load(f)
        wo = model.W_o.cpu().detach().numpy()

        readout_magnitude[r, 0] = (
            np.abs(wo[i_task_a]).mean() if sum(i_task_a) != 0 else 0
        )
        readout_magnitude[r, 1] = (
            np.abs(wo[i_task_b]).mean() if sum(i_task_b) != 0 else 0
        )
        readout_magnitude[r, 2] = np.abs(
            wo[~((i_task_a is True) | (i_task_b is True))]
        ).mean()
    readout_magnitude_int = readout_magnitude

    readout_magnitude = np.empty((50, 3))
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)

        yh = np.array(results["all_y_hidden"][1, :, :],dtype=np.float64)
        selectivity_matrix = np.zeros((100, 6))
        for i_neuron in range(100):
            y_neuron = yh[:, i_neuron]
            lr = sm.OLS(zscore(y_neuron), dmat)
            regr_results = lr.fit()
            # if only a single regressor is significant, store that neuron's selectivity
            if np.sum(regr_results.tvalues > 1.96) == 1:
                selectivity_matrix[
                    i_neuron,
                    np.where(regr_results.tvalues == np.max(regr_results.tvalues))[0][
                        0
                    ],
                ] = 1
        i_task_a = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 1)
            & (selectivity_matrix[:, 2] == 0)
            & (selectivity_matrix[:, 3] == 0)
        )
        i_task_b = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 0)
            & (selectivity_matrix[:, 2] == 1)
            & (selectivity_matrix[:, 3] == 0)
        )

        with open(
            "../checkpoints/" + hebb_models[1] + "/run_" + str(r) + "/model.pkl", "rb"
        ) as f:
            model = pickle.load(f)

        wo = model.W_o.cpu().detach().numpy()

        readout_magnitude[r, 0] = (
            np.abs(wo[i_task_a]).mean() if sum(i_task_a) != 0 else 0
        )
        readout_magnitude[r, 1] = (
            np.abs(wo[i_task_b]).mean() if sum(i_task_b) != 0 else 0
        )
        readout_magnitude[r, 2] = np.abs(
            wo[~((i_task_a is True) | (i_task_b is True))]
        ).mean()

    readout_magnitude_blocked = readout_magnitude

    f, axs = plt.subplots(1, 1, figsize=(2, 2), dpi=300)

    b1 = axs.bar(
        0 - 0.2,
        readout_magnitude_blocked[:, 0].mean(0),
        yerr=np.std(readout_magnitude_blocked[:, 0], 0) / np.sqrt(50),
        color=(39 / 255, 140 / 255, 145 / 255),
        width=0.2,
    )
    b2 = axs.bar(
        0,
        readout_magnitude_blocked[:, 1].mean(0),
        yerr=np.std(readout_magnitude_blocked[:, 1], 0) / np.sqrt(50),
        color=(34 / 255, 76 / 255, 128 / 255),
        width=0.2,
    )
    b3 = axs.bar(
        0 + 0.2,
        readout_magnitude_blocked[:, 2].mean(0),
        yerr=np.std(readout_magnitude_blocked[:, 2], 0) / np.sqrt(50),
        color=(159 / 255, 45 / 255, 235 / 255),
        width=0.2,
    )

    b1 = axs.bar(
        1 - 0.2,
        readout_magnitude_int[:, 0].mean(0),
        yerr=np.std(readout_magnitude_int[:, 0], 0) / np.sqrt(20),
        color=(39 / 255, 140 / 255, 145 / 255),
        width=0.2,
    )
    b2 = axs.bar(
        1,
        readout_magnitude_int[:, 1].mean(0),
        yerr=np.std(readout_magnitude_int[:, 1], 0) / np.sqrt(20),
        color=(34 / 255, 76 / 255, 128 / 255),
        width=0.2,
    )
    b3 = axs.bar(
        1 + 0.2,
        readout_magnitude_int[:, 2].mean(0),
        yerr=np.std(readout_magnitude_int[:, 2], 0) / np.sqrt(20),
        color=(159 / 255, 45 / 255, 235 / 255),
        width=0.2,
    )
    axs.set_xticks((0, 1))
    axs.set_xticklabels(("blocked", "interleaved"), rotation=0, fontsize=6)
    # ticks = np.round(axs.get_yticks(),2)
    # axs.set_yticklabels(ticks,fontsize=6)
    plt.yticks(fontsize=6)
    axs.set_title("Readout Weights", fontsize=6)
    axs.set_ylabel(r"weight magnitude (a.u.)", fontsize=6)
    axs.legend(
        [b1, b2, b3],
        ["1st task", "2nd task", "task agnostic"],
        fontsize=6,
        frameon=False,
    )

    res = ttest_ind(
        readout_magnitude_int[:, 0].ravel(), readout_magnitude_int[:, 2].ravel()
    )
    z = res.statistic
    print(f"int 1st vs agnostic: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([0.8, 1.2], [0.22, 0.22], "k-", linewidth=1)
    plt.text(1, 0.22, sigstar, ha="center", fontsize=6)

    res = ttest_ind(
        readout_magnitude_int[:, 1].ravel(), readout_magnitude_int[:, 2].ravel()
    )
    z = res.statistic
    print(f"int 2nd vs agnostic: t={z:.2f}, p={res.pvalue:.4f}")
    if res.pvalue >= 0.05:
        sigstar = "n.s."
    elif res.pvalue < 0.001:
        sigstar = "*" * 3
    elif res.pvalue < 0.01:
        sigstar = "*" * 2
    elif res.pvalue < 0.05:
        sigstar = "*"
    plt.plot([1, 1.2], [0.19, 0.19], "k-", linewidth=1)
    plt.text(1.1, 0.19, sigstar, ha="center", fontsize=6)
    sns.despine()
    plt.tight_layout()


def plot_gridsearch_modelvalidation(
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
    sluggish_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
):
    _, _, mats = eval.gen_behav_models()
    tmp_b = mats[0].ravel()[:, np.newaxis]
    tmp_i = mats[1].ravel()[:, np.newaxis]
    plt.subplot(1, 2, 1)
    plt.imshow(tmp_b.reshape(10, 5))
    plt.title("blocked model")
    plt.subplot(1, 2, 2)
    plt.imshow(tmp_i.reshape(10, 5))
    plt.title("interleaved model")

    mses_b = choicemodel.gridsearch_modelparams(
        tmp_b,
        curriculum="blocked",
        model_int=model_int,
        model_blocked=model_blocked,
        sluggish_vals=sluggish_vals,
        temp_vals=temp_vals,
    )
    plt.figure()
    plt.imshow(np.fliplr(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals))))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("blocked model")
    plt.colorbar()
    mses_i = choicemodel.gridsearch_modelparams(
        tmp_i,
        curriculum="interleaved",
        model_int=model_int,
        model_blocked=model_blocked,
        sluggish_vals=sluggish_vals,
        temp_vals=temp_vals,
    )
    plt.figure()
    plt.imshow(np.fliplr(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals))))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("interleaved model")
    plt.colorbar()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(1), color="lightgreen", linestyle="-"
    )
    plt.plot(
        np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(1),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(1)
            == np.min(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(1))
        )[0][0],
        np.min(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(1)),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(1)
            == np.min(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(1))
        )[0][0],
        np.min(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(1)),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.legend(("blocked", "interleaved"))
    plt.xlabel("slope")
    plt.ylabel("mse")
    plt.subplot(1, 2, 2)
    plt.plot(
        np.flip(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(0)),
        color="lightgreen",
        linestyle="-",
    )
    plt.plot(
        np.flip(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(0)),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            np.flip(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(0))
            == np.min(np.flip(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(0)))
        )[0][0],
        np.min(np.flip(np.array(mses_b).reshape(len(temp_vals), len(sluggish_vals)).mean(0))),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            np.flip(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(0))
            == np.min(np.flip(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(0)))
        )[0][0],
        np.min(np.flip(np.array(mses_i).reshape(len(temp_vals), len(sluggish_vals)).mean(0))),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.legend(("blocked", "interleaved"))
    plt.xlabel("sluggishness")
    plt.ylabel("mse")


def plot_gridsearch_singlesubjects(
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
    sluggish_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
):
    # grid search at single subject level
    gs_results = choicemodel.wrapper_gridsearch_modelparams(
        model_int=model_int,
        model_blocked=model_blocked,
        sluggish_vals=sluggish_vals,
        temp_vals=temp_vals,
    )
    plt.figure()
    plt.imshow(np.fliplr(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0)))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("single subject lvl, blocked")
    plt.colorbar()

    plt.figure()
    plt.imshow(np.fliplr(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0)))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("single subject lvl, interleaved")
    plt.colorbar()

    # averages
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1),
        color="lightgreen",
        linestyle="-",
    )
    plt.plot(
        gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)
            == np.min(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1))
        )[0][0],
        np.min(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)
            == np.min(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1))
        )[0][0],
        np.min(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.title("sigmoid slope")
    plt.xlabel("param val")
    plt.ylabel("mse")
    plt.legend(("blocked", "interleaved"))
    plt.subplot(1, 2, 2)
    plt.plot(
        np.flip(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)),
        color="lightgreen",
        linestyle="-",
    )
    plt.plot(
        np.flip(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            np.flip(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            == np.min(np.flip(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)))
        )[0][0],
        np.min(np.flip(gs_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            np.flip(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            == np.min(np.flip(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)))
        )[0][0],
        np.min(np.flip(gs_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.title("sluggishness")
    plt.xlabel("param val")
    plt.ylabel("mse")
    plt.legend(("blocked", "interleaved"))
    plt.tight_layout()

    print(
        f"estimated sluggishness (idx), interleaved:"
        f"{np.argmin(gs_results['cmat_i'].reshape(-1,20,20).mean(0).mean(0))}"
    )
    print(
        f"estimated sluggishness (idx), blocked: {np.argmin(gs_results['cmat_b'].reshape(-1,len(sluggish_vals),len(temp_vals)).mean(0).mean(0))}"
    )
    print(
        f"estimated slope (idx), interleaved: {np.argmin(gs_results['cmat_i'].reshape(-1,len(sluggish_vals),len(temp_vals)).mean(0).mean(1))}"
    )
    print(
        f"estimated slope (idx), blocked: {np.argmin(gs_results['cmat_b'].reshape(-1,len(sluggish_vals),len(temp_vals)).mean(0).mean(1))}"
    )
    # np.logspace(np.log(0.1), np.log(4), 20)[12]


def plot_gridsearch_group(
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
    sluggish_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
):
    # grid search at group level
    gs_m_results = choicemodel.wrapper_gridsearch_modelparams(
        single_subs=False,
        model_int=model_int,
        model_blocked=model_blocked,
        sluggish_vals=sluggish_vals,
        temp_vals=temp_vals,
    )
    plt.figure()
    plt.imshow(np.fliplr(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0)))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("single subject lvl, blocked")
    plt.colorbar()

    plt.figure()
    plt.imshow(np.fliplr(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0)))
    plt.xlabel("sluggishness")
    plt.ylabel("slope")
    plt.title("single subject lvl, interleaved")
    plt.colorbar()

    # averages
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1),
        color="lightgreen",
        linestyle="-",
    )
    plt.plot(
        gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)
            == np.min(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1))
        )[0][0],
        np.min(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)
            == np.min(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1))
        )[0][0],
        np.min(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(1)),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.title("sigmoid slope")
    plt.xlabel("param val")
    plt.ylabel("mse")
    plt.legend(("blocked", "interleaved"))
    plt.subplot(1, 2, 2)
    plt.plot(
        np.flip(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)),
        color="lightgreen",
        linestyle="-",
    )
    plt.plot(
        np.flip(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0)),
        color=[255 / 255, 102 / 255, 0],
        linestyle="-",
    )
    plt.scatter(
        np.where(
            np.flip(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            == np.min(
                np.flip(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            )
        )[0][0],
        np.min(np.flip(gs_m_results["cmat_b"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))),
        marker="d",
        s=100,
        color="lightgreen",
    )
    plt.scatter(
        np.where(
            np.flip(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            == np.min(
                np.flip(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))
            )
        )[0][0],
        np.min(np.flip(gs_m_results["cmat_i"].reshape(-1, len(temp_vals), len(sluggish_vals)).mean(0).mean(0))),
        marker="d",
        s=100,
        color=[255 / 255, 102 / 255, 0],
    )
    plt.title("sluggishness")
    plt.xlabel("param val")
    plt.ylabel("mse")
    plt.legend(("blocked", "interleaved"))
    plt.tight_layout()

    print(
        f"estimated sluggishness (idx), interleaved: {np.argmin(gs_m_results['cmat_i'].reshape(len(temp_vals), len(sluggish_vals)).mean(0))}"
    )
    print(
        f"estimated sluggishness (idx), blocked: {np.argmin(gs_m_results['cmat_b'].reshape(len(temp_vals), len(sluggish_vals)).mean(0))}"
    )
    print(
        f"estimated slope (idx), interleaved: {np.argmin(gs_m_results['cmat_i'].reshape(len(temp_vals), len(sluggish_vals)).mean(1))}"
    )
    print(
        f"estimated slope (idx), blocked: {np.argmin(gs_m_results['cmat_b'].reshape(len(temp_vals), len(sluggish_vals)).mean(1))}"
    )
