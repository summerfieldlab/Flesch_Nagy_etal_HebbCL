import pickle
from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import ttest_ind


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
            sigstar = helper_ttest(p_blocked, p_interleaved)

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
    """
    creates a colormap and returns both the cmap object
    and a list of rgb tuples
    inputs:
    -basecols: nump array with rgb colors spanning the cmap
    -n_items: sampling resolution of cmap
    outputs:
    -cmap: the cmap object
    -cols: np array of colors spanning cmap
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
    """
    n_items: number of items along one dimension covered by grid
    """
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
    _ = plt.plot([wi[-1] for wi in ws], color="blue")
    _ = plt.plot([wi[-2] for wi in ws], color="orange")
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
    _ = plt.plot([wi[-1] for wi in ws], color="blue")
    _ = plt.plot([wi[-2] for wi in ws], color="orange")
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
    a = plt.plot([wi[-2] for wi in ws], color="blue")
    b = plt.plot([wi[-1] for wi in ws], color="orange")
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
    plt.plot(ta, color="blue")
    plt.plot(tb, color="orange")
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
):
    """plots learning curves (acc, task selectivity, context corr) and choice mats

    Args:
        n_runs (int, optional): _description_. Defaults to 50.
        n_epochs (int, optional): _description_. Defaults to 200.
        models (list, optional): _description_. Defaults to ['baseline_interleaved_new_select',
        'baseline_blocked_new_select'].
    """

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
                cc = np.clip(results["all_y_out"][1, :], -709.78, 709.78).astype(
                    np.float64
                )
                choices = 1 / (1 + np.exp(-cc))
                cmats_a.append(choices[:25].reshape(5, 5))
                cmats_b.append(choices[25:].reshape(5, 5))

        cmats_a = np.array(cmats_a)
        cmats_b = np.array(cmats_b)

        # accuracy
        axs1[i].plot(np.arange(n_epochs), acc_1st.mean(0), color="orange")
        axs1[i].fill_between(
            np.arange(n_epochs),
            acc_1st.mean(0) - np.std(acc_1st, 0) / np.sqrt(n_runs),
            acc_1st.mean(0) + np.std(acc_1st, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color="orange",
            edgecolor=None,
        )
        axs1[i].plot(np.arange(n_epochs), acc_2nd.mean(0), color="blue")
        axs1[i].fill_between(
            np.arange(n_epochs),
            acc_2nd.mean(0) - np.std(acc_2nd, 0) / np.sqrt(n_runs),
            acc_2nd.mean(0) + np.std(acc_2nd, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color="blue",
            edgecolor=None,
        )
        axs1[i].set_ylim([0.4, 1.05])
        axs1[i].set(xlabel="trial", ylabel="accuracy")
        axs1[i].legend(["1st task", "2nd task"], frameon=False)
        if "interleaved" not in m:
            axs1[i].plot([100, 100], [0, 1], "k--", alpha=0.5)
        axs1[i].set_title(m.split("_")[1])
        plt.gcf()
        sns.despine(f1)
        f1.tight_layout()

        # unit allocation (task factorisation)
        axs2[i].plot(np.arange(n_epochs), t_b.mean(0), color="orange")
        axs2[i].fill_between(
            np.arange(n_epochs),
            t_b.mean(0) - np.std(t_b, 0) / np.sqrt(n_runs),
            t_b.mean(0) + np.std(t_b, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color="orange",
            edgecolor=None,
        )
        axs2[i].plot(np.arange(n_epochs), t_a.mean(0), color="blue")
        axs2[i].fill_between(
            np.arange(n_epochs),
            t_a.mean(0) - np.std(t_a, 0) / np.sqrt(n_runs),
            t_a.mean(0) + np.std(t_a, 0) / np.sqrt(n_runs),
            alpha=0.5,
            color="blue",
            edgecolor=None,
        )
        axs2[i].set_yticks([0, 0.5, 1])
        ticks = axs2[i].get_yticks()  # plt.yticks()
        axs2[i].set_yticklabels((int(x) for x in ticks * 100))
        axs2[i].set(xlabel="trial", ylabel="task-sel (%)")
        axs2[i].legend(["1st task", "2nd task"], frameon=False)
        if "interleaved" not in m:
            axs2[i].plot([100, 100], [0, 1], "k--", alpha=0.5)
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
            axs3[i].plot([100, 100], [-1, 1], "k--", alpha=0.5)
        axs3[i].set_title(m.split("_")[1])
        sns.despine(f3)
        f3.tight_layout()

        # choice matrices
        axs4[i, 0].imshow(cmats_a.mean(0))
        axs4[i, 0].set_title("1st task")
        axs4[i, 0].set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="irrel", ylabel="rel")
        axs4[i, 1].imshow(cmats_b.mean(0))
        axs4[i, 1].set(xticks=[0, 2, 4], yticks=[0, 2, 4], xlabel="rel", ylabel="irrel")
        axs4[i, 1].set_title("2nd task")
        PCM = axs4[i, 1].get_children()[
            -2
        ]  # get the mappable, the 1st and the 2nd are the x and y axes

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(PCM, cax=cax)

    f1.tight_layout()
    f2.tight_layout()
    f3.tight_layout()
    f4.tight_layout()
