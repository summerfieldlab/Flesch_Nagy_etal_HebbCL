import numpy as np
import torch
import math
from utils.nnet import from_gpu
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore, multivariate_normal
from sklearn.linear_model import LinearRegression
from scipy.io import loadmat




def compute_behav_taskfact(cmats_dict: dict) -> dict:
    """performs regression to compute behavioural task factorisation

    Args:
        cmats_dict (dict): dictionary with choice matrices

    Returns:
        dict: dictionary with estimated beta coefficients
    """
    _, dmat, _ = gen_behav_models()
    betas_dict = {}
    betas = []
    for r in np.arange(0, len(cmats_dict["int_a"])):
        choices = np.concatenate(
            (
                cmats_dict["int_a"][r, :, :].flatten(),
                cmats_dict["int_b"][r, :, :].flatten(),
            ),
            axis=0,
        )[:, np.newaxis]
        assert choices.shape == (50, 1)
        yrdm = squareform(pdist(choices))

        y = zscore(yrdm[np.tril_indices(50, k=-1)]).flatten()
        assert len(y) == 1225
        lr = LinearRegression()
        lr.fit(dmat, y)
        betas.append(lr.coef_)
    betas_dict["int"] = np.array(betas)

    betas = []
    for r in np.arange(0, len(cmats_dict["bloc_a"])):
        choices = np.concatenate(
            (
                cmats_dict["bloc_a"][r, :, :].flatten(),
                cmats_dict["bloc_b"][r, :, :].flatten(),
            ),
            axis=0,
        )[:, np.newaxis]
        assert choices.shape == (50, 1)
        yrdm = squareform(pdist(choices))

        y = zscore(yrdm[np.tril_indices(50, k=-1)]).flatten()
        assert len(y) == 1225
        lr = LinearRegression()
        lr.fit(dmat, y)
        betas.append(lr.coef_)
    betas_dict["bloc"] = np.array(betas)
    return betas_dict


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


def compute_congruency_acc(cmat, cmat_true, n_samp=10000):
    """computes accuracy on congruent and incongruent trials

    Args:
        cmat (np array): choices
        cmat_true (np array): ground truth category labels
        n_samp (int, optional): number of samples. Default to 10000

    Returns:
        int: accuracies on congruent and incongruent trials
    """

    c = np.array(
        [(cmat > np.random.rand(*cmat.shape)) == cmat_true for _ in range(n_samp)]
    ).mean(0)

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
    # assert yout.shape == (100, 50)
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


def gen_modelrdms(ctx_offset=1):
    models = []
    ctx = np.concatenate(
        (ctx_offset * np.ones((25, 1)), np.zeros((25, 1))), axis=0
    ).reshape(50, 1)
    # model rdms:
    a, b = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
    # grid model
    gridm = np.concatenate(
        (a.flatten()[np.newaxis, :], b.flatten()[np.newaxis, :]), axis=0
    ).T
    gridm = np.concatenate((np.tile(gridm, (2, 1)), ctx), axis=1)
    models.append(squareform(pdist(gridm, metric="euclidean")))

    # # rotated grid model
    # gridm = np.concatenate((a.flatten()[np.newaxis,:],b.flatten()[np.newaxis,:]),axis=0).T
    # gridm[25:, :] = gridm[25:, :] @ np.array([[np.cos(np.deg2rad(270)), -np.sin(np.deg2rad(270))],
    # [np.sin(np.deg2rad(270)), np.cos(np.deg2rad(270))]])
    # gridm = np.concatenate((np.tile(gridm,(2,1)),ctx),axis=1)
    # models.append(squareform(pdist(gridm,metric='euclidean')))

    # orthogonal model
    orthm = np.concatenate(
        (
            np.concatenate((b.flatten()[np.newaxis, :], np.zeros((1, 25))), axis=0).T,
            np.concatenate((np.zeros((1, 25)), a.flatten()[np.newaxis, :]), axis=0).T,
        ),
        axis=0,
    )
    orthm = np.concatenate((orthm, ctx), axis=1)
    models.append(squareform(pdist(orthm, metric="euclidean")))

    # # parallel model
    # a = a.flatten()
    # b = b.flatten()

    # ta = np.stack((a,np.zeros((25))),axis=1)
    # tb = np.stack((np.zeros(25),b),axis=1)
    # theta = np.radians(-90)
    # c, s = np.cos(theta), np.sin(theta)
    # R = np.array(((c, -s), (s, c)))

    # parm = np.concatenate((ta.dot(R),tb),axis=0)
    # parm = np.concatenate((parm,ctx),axis=1)
    # models.append(squareform(pdist(parm,metric='euclidean')))

    # # only branchiness model
    # obm = np.concatenate((a[:,np.newaxis],a[:,np.newaxis]),axis=0)

    # models.append(squareform(pdist(obm,metric='euclidean')))

    # # only leafiness model
    # olm = np.concatenate((b[:,np.newaxis],b[:,np.newaxis]),axis=0)
    # models.append(squareform(pdist(olm,metric='euclidean')))

    # diagonal model
    a = a.flatten()
    b = b.flatten()
    ta = np.stack((a, b), axis=1)
    tb = np.stack((a, b), axis=1)
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    # R = np.array(((c, -s), (s, c)))
    r = np.array([(c, s)]).T
    assert r.shape == (2, 1)

    diagm = np.concatenate((ta @ r @ r.T, tb @ r @ r.T), axis=0)
    diagm = np.concatenate((diagm, ctx), axis=1)
    models.append(squareform(pdist(diagm, metric="euclidean")))

    # construct design matrix
    dmat = np.asarray(
        [zscore(rdm[np.tril_indices(50, k=-1)].flatten()) for rdm in models]
    ).T

    rdms = np.asarray(models)

    return rdms, dmat


def smooth(x: np.ndarray, w: int = 25):
    xs = np.empty((len(x), len(x.T) // w))

    for i in range(len(x)):
        jstart = 0
        for j in range((len(x.T) // w)):
            xs[i, j] = np.nanmean(x[i, jstart : jstart + w])
            jstart += w
    return xs


def lcurves_humandata(
    filepath: str = "D:/RA_PNAS/github_release/Data/Exp1a/allData/",
    filename: str = "allData_exp1a.mat",
):
    dataset = loadmat(filepath + filename, struct_as_record=False, squeeze_me=True)
    dataset = vars(dataset["allData"])
    ids_int = np.where(dataset["subCodes"][0, :] == 3)[0]
    ids_b200 = np.where(
        (dataset["subCodes"][0, :] != 3) & (dataset["subCodes"][4, :] == 200)
    )[0]

    # update acc vects
    training_accs = np.empty((dataset["resp_correct"].shape[1], 400))
    for i, s in enumerate(range(dataset["resp_correct"].shape[1])):
        # get cat and acc vectors
        c = dataset["expt_catIDX"][:400, i]
        a = dataset["resp_correct"][:400, i].astype(np.float32)
        # set boundary trials to nan:
        a[
            c == 0,
        ] = np.nan
        # store acc vect
        training_accs[i, :] = a

    # extract blocked and int groups:
    lcurves_b200 = smooth(training_accs[ids_b200, :], w=50)
    lcurves_int = smooth(training_accs[ids_int, :], w=50)
    return {"blocked": lcurves_b200, "interleaved": lcurves_int}
