import pickle
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.io import loadmat
from utils.eval import gen_behav_models
from typing import List
from scipy.optimize import minimize


def softmax(x: np.array, T=1e-3) -> np.array:
    """softmax non-linearity

    Args:
        x (np.array): inputs to softmax
        T (float, optional): softmax temperature. Defaults to 1e-3.

    Returns:
        np.array: outputs of softmax
    """

    return np.exp(x / T) / (np.exp(x / T) + np.exp(0 / T))


def choice_sigmoid(x: np.array, T=1e-3) -> np.array:
    """applies sigmoid to inputs

    Args:
        x (np.array): array of numbers
        T (float, optional): temperature (slope of sigmoid). Defaults to 1e-3.

    Returns:
        np.array: outputs of sigmoid
    """
    cc = np.clip(-x / T, -709.78, 709.78).astype(np.float64)
    sigm = 1.0 / (1.0 + np.exp(cc))
    return sigm


def mse(x: np.array, y: np.array) -> float:
    """mean squared error between two vectors

    Args:
        x (np.array): a vector of floats
        y (np.array): a vector of floats

    Returns:
        float: the mse between x and y
    """
    mse = np.mean(np.sqrt((x - y) ** 2))
    return mse


def compute_mse_choicemats(
    y_sub: np.array,
    i_slug: int,
    i_temp: int,
    slug_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
    n_runs=50,
    curriculum="interleaved",
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
) -> np.array:
    """computes mse between human and model choices

    Args:
        y_sub (np.array): choices made by a single human participant (task a & b concatenated)
        i_slug (int): index of sluggishness value to fit
        i_temp (int): index of temperature value to fit
        slug_vals (np.array, optional): sluggishness values to use. Defaults to np.round(np.linspace(0.05,1,20),2).
        temp_vals (np.array, optional): sigmoid temperature values to use.
            Defaults to np.logspace(np.log(0.1),np.log(4),20).
        n_runs (int, optional): number of runs to include. Defaults to 50
        curriculum (str, optional): training curriculum (blocked or interleaved). Defaults to interleaved

    Returns:
        np.array: mse for current fit
    """

    # load models with requested sluggishness value and average over outputs
    curric_str = model_int if curriculum == "interleaved" else model_blocked
    y_net = []
    for r in np.arange(0, n_runs):
        with open(
            "../checkpoints/"
            + curric_str
            + str(i_slug)
            + "/run_"
            + str(r)
            + "/results.pkl",
            "rb",
        ) as f:
            results = pickle.load(f)
            y = choice_sigmoid(results["all_y_out"][1, :, :], T=temp_vals[i_temp])
            y_net.append(y)
    y_net = np.array(y_net).mean(0)
    assert len(y_net) == 50

    # pass averaged outputs through sigmoid with chosen temp value
    # y_net = choice_sigmoid(y_net, T=temp_vals[i_temp])

    # compute and return mse between human and simulated model choices
    return mse(y_sub, y_net)


def gridsearch_modelparams(
    y_sub: np.array,
    n_jobs=-1,
    curriculum="interleaved",
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
    sluggish_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
) -> np.array:
    """performs grid search over softmax temperature and
       sluggishness param

    Args:
        y_sub (np.array): vector with choice probabilities of single subject
        n_jobs (int, optional): number of parallel jobs. Defaults to -1
        curriculum (str, optional): training curriculum (blocked or interleaved). Defaults to interleaved

    Returns:
        np.array: grid of MSE vals for each hp combination
    """

    idces_sluggishness = np.arange(0, len(sluggish_vals))
    idces_temp = np.arange(0, len(temp_vals))
    a, b = np.meshgrid(idces_sluggishness, idces_temp)
    a, b = a.flatten(), b.flatten()
    if n_jobs != 1:
        mses = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(compute_mse_choicemats)(
                y_sub,
                i_slug,
                i_temp,
                n_runs=20,
                curriculum=curriculum,
                model_int=model_int,
                model_blocked=model_blocked,
            )
            for i_slug, i_temp in zip(a, b)
        )
    else:
        mses = [
            compute_mse_choicemats(
                y_sub,
                i_slug,
                i_temp,
                n_runs=20,
                curriculum=curriculum,
                model_int=model_int,
                model_blocked=model_blocked,
                slug_vals=sluggish_vals,
                temp_vals=temp_vals,
            )
            for i_slug, i_temp in zip(a, b)
        ]
    return mses


def wrapper_gridsearch_modelparams(
    single_subs=True,
    model_int: str = "sluggish_oja_int_select_sv",
    model_blocked: str = "sluggish_oja_blocked_select_sv",
    sluggish_vals=np.round(np.linspace(0.05, 1, 20), 2),
    temp_vals=np.logspace(np.log(0.1), np.log(4), 20),
) -> dict:
    """wrapper for gridsearch of modelparams
    Args:
        single_subs (bool, optional): whether or not to fit to individual participants. Defaults to True
    Returns:
        dict: mse for blocked and interleaved groups
    """

    cmats = loadmat("../datasets/choicemats_exp1a.mat")
    keys = ["cmat_b", "cmat_i"]
    results = {}
    for k in keys:
        if k.split("_")[-1] == "b":
            curriculum = "blocked"
        elif k.split("_")[-1] == "i":
            curriculum = "interleaved"
        if single_subs:
            mses = []
            for c in range(len(cmats[k + "_north"])):
                y_sub = np.concatenate(
                    (
                        cmats[k + "_north"][c, :, :].ravel(),
                        cmats[k + "_south"][c, :, :].ravel(),
                    ),
                    axis=0,
                )[:, np.newaxis]
                assert len(y_sub) == 50
                mses.append(
                    gridsearch_modelparams(
                        y_sub,
                        curriculum=curriculum,
                        model_int=model_int,
                        model_blocked=model_blocked,
                        sluggish_vals=sluggish_vals,
                        temp_vals=temp_vals,
                    )
                )
        else:
            y_sub = np.concatenate(
                (
                    cmats[k + "_north"].mean(0).ravel(),
                    cmats[k + "_south"].mean(0).ravel(),
                ),
                axis=0,
            )[:, np.newaxis]
            assert len(y_sub) == 50
            mses = gridsearch_modelparams(
                y_sub,
                curriculum=curriculum,
                model_int=model_int,
                model_blocked=model_blocked,
                sluggish_vals=sluggish_vals,
                temp_vals=temp_vals,
            )
        results[k] = np.asarray(mses)
    return results


def sample_choices(y_est: np.array, n_samp=10000) -> np.array:
    """samples choices from sigmoidal inputs

    Args:
        y_est (np.array): array with choice probabilities
        n_samp (int, optional): number of samples to draw. Defaults to 1000.

    Returns:
        np.array: sampled choices
    """
    sampled_choices = np.array(
        [y_est > np.random.rand(*y_est.shape) for i in range(n_samp)]
    )
    return sampled_choices


# sampled_choices = sample_choices(cmats_a)

def fit_model_to_subjects(choicemats, n_runs=1):
    """
    wrapper for fit_choice_model
    loops over subjects
    """
    tasks = ["task_a", "task_b"]

    thetas = {"bias_a": [], "bias_b": [], "lapse": [], "slope": [], "offset": []}
    for sub in range(len(choicemats[tasks[0]])):
        cmat_a = choicemats[tasks[0]][sub, :, :]
        cmat_b = choicemats[tasks[1]][sub, :, :]
        cmats = np.concatenate((cmat_a.flatten(), cmat_b.flatten()))
        theta_hat = fit_choice_model(cmats, n_runs=n_runs)
        theta_hat[0] = angular_bias(theta_hat[0], 90, task="a")
        theta_hat[1] = angular_bias(theta_hat[1], 180, task="b")
        for idx, k in enumerate(thetas.keys()):
            thetas[k].append(theta_hat[idx])
    for k in thetas.keys():
        thetas[k] = np.asarray(thetas[k])
    thetas["bias"] = np.stack((thetas["bias_a"], thetas["bias_b"]), axis=1).mean(1)
    return thetas

def compute_sampled_accuracy(cmat_a: np.array, cmat_b: np.array, flipdims: bool =False) -> float:
    """computes accuracy based on choices sampled from sigmoid

    Args:
        cmat_a (np.array): estimated choice matrix for task a
        cmat_b (np.array): esimated choie matrix for task b

    Returns:
        float: computed accuracy
    """
    # generate ground truth matrices:
    _, _, cmats = gen_behav_models()
    if flipdims:
        cmat_gt_a = cmats[0, 1, :, :]
        cmat_gt_b = cmats[0, 0, :, :]
    else:
        cmat_gt_a = cmats[0, 0, :, :]
        cmat_gt_b = cmats[0, 1, :, :]
    # indices of non-boundary trials:
    valid_a = np.where(cmat_gt_a.ravel() != 0.5)
    valid_b = np.where(cmat_gt_b.ravel() != 0.5)
    # sample choices
    cmat_samp_a = sample_choices(cmat_a)
    cmat_samp_b = sample_choices(cmat_b)
    # calculate accuracy for each sample, excluding boundary trials
    accs_a = [
        np.mean(cmat_gt_a.ravel()[valid_a] == samp.ravel()[valid_a])
        for samp in cmat_samp_a
    ]
    accs_b = [
        np.mean(cmat_gt_b.ravel()[valid_b] == samp.ravel()[valid_b])
        for samp in cmat_samp_b
    ]
    return (np.mean(accs_a) + np.mean(accs_b)) / 2


def sigmoid(x: np.float, L: np.float, k: np.float, x0: np.float) -> np.float:
    """sigmoidal non-linearity with three free parameters
        note: x can also be a vector
    Args:
        x (np.float): inputs
        L (np.float): lapse rate (0,0.5)
        k (np.float): slope (0,)
        x0 (np.float): offset (0,)

    Returns:
        np.float: transformed y-value
    """

    y = L + (1 - L * 2) / (1.0 + np.exp(-k * (x - x0)))
    return y


def scalar_projection(X: np.array, phi: np.float) -> np.float:
    """performs scalar projection of x onto y by angle phi

    Args:
        X (np.array): inputs
        phi (np.float): angle of projection

    Returns:
        np.float: projected values
    """
    phi_bound = np.deg2rad(phi)
    phi_ort = phi_bound - np.deg2rad(90)
    y = X @ np.array([np.cos(phi_ort), np.sin(phi_ort)]).T
    return y


def angular_distance(target_ang: np.float, source_ang: np.float) -> np.float:
    """computes angular distance between source and target angle

    Args:
        target_ang (np.float): angle in degrees
        source_ang (np.float): angle in degrees

    Returns:
        np.float: angular distance in degrees
    """
    target_ang = np.deg2rad(target_ang)
    source_ang = np.deg2rad(source_ang)
    return np.rad2deg(
        np.arctan2(np.sin(target_ang - source_ang), np.cos(target_ang - source_ang))
    )


def angular_bias(ref: np.float, est: np.float, task="a") -> np.float:
    """computes angular bias (0= orth,45= diag bounds)

    Args:
        ref (np.float): reference angle
        est (np.float): estimated angle
        task (str, optional): which task (determines sign). Defaults to "a".

    Returns:
        np.float: angular bias
    """
    bias = angular_distance(est, ref)
    if task == "a":
        bias = -bias
    return bias


def objective_function(X: np.array, y_true: np.array) -> np.float:
    """computes loss between labels and predictions

    Args:
        X (np.array): inputs to model
        y_true (np.array): labels

    Returns:
        np.float: loss
    """

    def loss(theta):
        return -np.sum(np.log(1.0 - np.abs(y_true - choice_model(X, theta)) + 1e-5))

    return loss


def choice_model(X: np.array, theta: List[np.float]) -> np.array:
    """generates choice probability matrix
    free parameters: orientation of bound, slope, offset and lapse rate of sigmoidal transducer

    Args:
        X (np.array): inputs
        theta (List[np.float]): parameters (projection angle a & b, lapse, slope and offset of sigmoid)

    Returns:
        np.array: predictions
    """

    # projection task a
    X1 = scalar_projection(X, theta[0])
    # projection task b
    X2 = scalar_projection(X, theta[1])

    # inputs to model
    X_in = np.concatenate((X1, X2))

    # pass through transducer:
    y_hat = sigmoid(X_in, theta[2], theta[3], theta[4])

    # return outputs
    return y_hat


def fit_choice_model(y_true: np.array, n_runs=1) -> List:
    """fits choice model to data, using Nelder-Mead or L-BFGS-B algorithm

    Args:
        y_true (np.array): labels
        n_runs (int, optional): number of runs. Defaults to 1.

    Returns:
        List: estimated parameters
    """

    assert n_runs > 0

    a, b = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
    a = a.flatten()
    b = b.flatten()
    X = np.stack((a, b)).T
    theta_bounds = ((0, 360), (0, 360), (0, 0.5), (0, 20), (-1, 1))
    if n_runs == 1:
        theta_init = [90, 180, 0, 2, 0]
        results = minimize(
            objective_function(X, y_true),
            theta_init,
            bounds=theta_bounds,
            method="L-BFGS-B",
        )
        return results.x
    elif n_runs > 1:
        theta_initbounds = ((80, 100), (170, 190), (0, 0.1), (14, 16), (-0.02, 0.02))
        thetas = []
        for i in range(10):
            theta_init = [
                np.round(np.random.uniform(a[0], a[1]), 2) for a in theta_initbounds
            ]
            results = minimize(
                objective_function(X, y_true),
                theta_init,
                bounds=theta_bounds,
                method="L-BFGS-B",
            )
            thetas.append(results.x)

        return np.mean(np.array(thetas), 0)

# scratchpad


def nolapse(func):
    """decorator for sigmoid_fourparmas to avoid fitting lapse rate"""

    def inner(x, L, k, x0):
        return func(x, L, k, x0, fitlapse=False)

    return inner


def sigmoid_fourparams(
    x: np.array, L: float, k: float, x0: float, fitlapse=True
) -> np.array:
    """sigmoidal non-linearity with four free params

    Args:
        x (np.array): inputs
        L (float): lapse rate
        k (float): slope
        x0 (float): offset
        fitlapse (bool, optional): fit lapse rate. Defaults to True.

    Returns:
        np.array: outputs of sigmoid
    """
    if fitlapse is False:
        L = 0
    y = L + (1 - L * 2) / (1.0 + np.exp(-k * (x - x0)))
    return y


def fit_sigmoid(x: np.array, y, fitlapse=True):
    """
    fits sigmoidal nonlinearity to some data
    returns best-fitting parameter estimates
    """
    # initial guesses for max, slope and inflection point
    theta0 = [0.0, 0.0, 0.0]
    if fitlapse is False:
        popt, _ = curve_fit(
            nolapse(sigmoid_fourparams),
            x,
            y,
            theta0,
            method="dogbox",
            maxfev=1000,
            bounds=([0, -10, -10], [0.5, 10, 10]),
        )
    else:
        popt, _ = curve_fit(
            sigmoid_fourparams,
            x,
            y,
            theta0,
            method="dogbox",
            maxfev=1000,
            bounds=([0, -10, -10], [0.5, 10, 10]),
        )

    return popt
