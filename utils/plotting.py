import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
import seaborn as sns


def disp_model_estimates(thetas, cols=[[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]):
    """
    displays average parameter estimates of choice model
    """
    sem = lambda x, ax: np.nanstd(x, ddof=1, axis=ax) / np.sqrt(np.shape(x)[ax])
    parameters = ["bias", "lapse", "slope", "offset"]
    curricula = ["blocked", "interleaved"]

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

        # _, pval = mannwhitneyu(p_blocked, p_interleaved)
        # ax.set(
        #     xticks=[0, 1],
        #     title=param + ", p=" + str(np.round(pval, 3)),
        # )

        ax.set(
            xticks=[0, 1],
            title=param,
        )
        ax.set_xticklabels(("blocked", "interleaved"), rotation=90)
        if param == "bias":
            ax.set_ylabel("angular bias (°)")
            ax.set_ylim((0, 30))
        elif param == "lapse":
            ax.set_ylabel("lapses (%)")
            ax.set_ylim((0, 0.2))
            ax.set_yticklabels(np.round(ax.get_yticks() * 100, 2))
        elif param == "slope":
            ax.set_ylabel("slope (a.u)")
        elif param == "offset":
            ax.set_ylabel("offset (a.u.)")
        sns.despine()

        if param == "slope":
            ax.set_ylim((0, 15))
        elif param == "offset":
            ax.set_ylim((-0.05, 0.2))

    plt.rc("figure", titlesize=6)
    plt.tight_layout()


def helper_make_colormap(
    basecols=np.array(
        [[63, 39, 24], [64, 82, 21], [65, 125, 18], [66, 168, 15], [68, 255, 10]]
    )
    / 255,
    n_items=5,
    monitor=False,
):
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


def plot_grid2(xy, line_colour="green", line_width=1, fig_id=1, n_items=5):
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
    fig = plt.figure(fig_id)

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
    xyz,
    task_id="a",
    fig_id=1,
    flipdims=False,
    items_per_dim=5,
    flipcols=False,
    marker_scale=1,
):

    if flipcols == True:
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
    itemMarkerSize = 1
    scat_b = np.linspace(0.5, 2.5, items_per_dim) * marker_scale
    # # scat_l = np.array([[3,252,82], [3,252,177], [3,240,252], [3,152,252], [3,73,252]])/255
    # scat_l = np.array([[63,39,24], [64,82,21], [65,125,18], [66,168,15], [68,255, 10]])/255
    # # scat_l = np.array([[0,0,0], [.2,.2,.2],[.4,.4,.4],[.6,.6,.6],[.8,.8,.8]])
    _, scat_l = helper_make_colormap(
        basecols=np.array(
            [[63, 39, 24], [64, 82, 21], [65, 125, 18], [66, 168, 15], [68, 255, 10]]
        )
        / 255,
        n_items=items_per_dim,
        monitor=False,
    )

    b, l = np.meshgrid(np.arange(0, items_per_dim), np.arange(0, items_per_dim))
    if flipdims == True:
        l, b = np.meshgrid(np.arange(0, items_per_dim), np.arange(0, items_per_dim))

    b = b.flatten()
    l = l.flatten()
    x = xyz[:, 0]
    y = xyz[:, 1]

    if task_id == "both":
        b = np.concatenate((b, b), axis=0)
        l = np.concatenate((l, l), axis=0)
        fig = plt.figure(fig_id)

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
    embedding, fig, fig_id=2, axlims=2.5, flipdims=False, monk=False, flipcols=False
):

    # if flipdims==True:
    #     col1 = (0, 0, .5)
    #     col2 = (255/255,102/255,0)

    if flipcols == True:
        col1 = (0, 0, 0.5)
        col2 = (255 / 255, 102 / 255, 0)
    else:
        col1 = (255 / 255, 102 / 255, 0)
        col2 = (0, 0, 0.5)

    if monk == True:
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


def plot_results(ws, delta_ws, n_trials, eta, sigma):

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
    a = plt.plot([wi[-1] for wi in ws], color="blue")
    b = plt.plot([wi[-2] for wi in ws], color="orange")
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


def plot_initsign_results(ws, delta_ws, n_trials, eta, init_weights=[1, -1]):

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
    a = plt.plot([wi[-1] for wi in ws], color="blue")
    b = plt.plot([wi[-2] for wi in ws], color="orange")
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


def plot_ghasla_results(data, ws, delta_ws, n_trials, eta, sigma):

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
        ya = np.maximum((data["x_task_a"] @ W), 0).mean(0)
        yb = np.maximum((data["x_task_b"] @ W), 0).mean(0)
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
