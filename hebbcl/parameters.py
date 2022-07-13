import argparse
from typing import Union


def boolean_string(s: str) -> bool:
    """helper function, turns string into boolean variable

    Args:
        s (str): input string

    Raises:
        ValueError: if s neither False nor True

    Returns:
        bool: s turned into boolean
    """
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def none_or_str(value: str) -> Union[str, None]:
    """helper function, turns "None" into proper None

    Args:
        value (str): a string

    Returns:
        Union[str, None]: None if input is "None" or "none"
    """
    if value == "None" or value == "none":
        return None
    return value


def set_hpo_args(
    args: argparse.Namespace, whichmodel: str = "interleaved_vanilla_1ctx"
) -> argparse.Namespace:
    """sets args to specific values depending on target model and curriculum

    Args:
        args (argparse.Namespace): parsed arguments
        whichmodel (str, optional): target model & curriculum.
        Defaults to "interleaved_vanilla_1ctx".

    Returns:
        argparse.Namespace: modified arguments
    """

    args.hpo_fixedseed = True
    args.hpo_scheduler = "bohb"
    args.hpo_searcher = "bohb"

    if whichmodel == "interleaved_vanilla_1ctx":
        args.ctx_twice = False
        args.training_schedule = "interleaved"
        args.perform_hebb = False
        args.centering = True
        args.gating = None

    elif whichmodel == "blocked_vanilla_1ctx":
        args.ctx_twice = False
        args.training_schedule = "blocked"
        args.perform_hebb = False
        args.centering = False
        args.gating = None

    elif whichmodel == "interleaved_vanilla_2ctx":
        args.ctx_twice = True
        args.training_schedule = "interleaved"
        args.perform_hebb = False
        args.centering = False
        args.gating = None

    elif whichmodel == "blocked_vanilla_2ctx":
        args.ctx_twice = True
        args.training_schedule = "blocked"
        args.perform_hebb = False
        args.centering = False
        args.gating = None

    elif whichmodel == "interleaved_ojactx_1ctx":
        args.ctx_twice = False
        args.training_schedule = "interleaved"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja_ctx"

    elif whichmodel == "blocked_ojactx_1ctx":
        args.ctx_twice = False
        args.training_schedule = "blocked"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja_ctx"

    elif whichmodel == "interleaved_ojactx_2ctx":
        args.ctx_twice = True
        args.training_schedule = "interleaved"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja_ctx"

    elif whichmodel == "blocked_ojactx_2ctx":
        args.ctx_twice = True
        args.training_schedule = "blocked"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja_ctx"

    elif whichmodel == "interleaved_ojaall_1ctx":
        args.ctx_twice = False
        args.training_schedule = "interleaved"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja"

    elif whichmodel == "blocked_ojaall_1ctx":
        args.ctx_twice = False
        args.training_schedule = "blocked"
        args.perform_hebb = True
        args.centering = True
        args.gating = "oja"

    else:
        raise ValueError("requested config not available")
    return args


# parameters
parser = argparse.ArgumentParser(description="Hebbian Continual Learning simulations")

# data parameters
parser.add_argument(
    "--ctx_scaling", default=2, type=float, help="scaling of context signal"
)
parser.add_argument(
    "--ctx_avg", default=True, type=boolean_string, help="context averaging"
)
parser.add_argument(
    "--ctx_avg_type", default="ema", type=str, help="avg type (ema or sma)"
)
parser.add_argument(
    "--ctx_avg_window", default=50, type=int, help="ctx avg window width (sma)"
)
parser.add_argument(
    "--ctx_avg_alpha",
    default=1,
    type=float,
    help="ctx avg smoothing parameter alpha (ema)",
)
parser.add_argument(
    "--centering", default="True", type=boolean_string, help="centering of data"
)

# network parameters
parser.add_argument("--n_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument(
    "--n_features", default=27, type=int, help="number of stimulus units"
)
parser.add_argument("--n_out", default=1, type=int, help="number of output units")
parser.add_argument("--n_hidden", default=100, type=int, help="number of hidden units")
parser.add_argument(
    "--weight_init", default=1e-2, type=float, help="initial input weight scale"
)
parser.add_argument(
    "--ctx_w_init", default=1e-2, type=float, help="initial context weight scale"
)

parser.add_argument(
    "--ctx_twice",
    default=False,
    type=boolean_string,
    help="apply context to both hidden layers",
)
# optimiser parameters
parser.add_argument(
    "--lrate_sgd", default=1e-2, type=float, help="learning rate for SGD"
)
parser.add_argument(
    "--lrate_hebb", default=0.01, type=float, help="learning rate for hebbian update"
)
parser.add_argument(
    "--hebb_normaliser",
    default=10.0,
    type=float,
    help="normalising const. for hebbian update",
)
parser.add_argument(
    "--gating",
    default="oja_ctx",
    type=none_or_str,
    help="any of: None, manual, GHA, SLA, oja, oja_ctx",
)
parser.add_argument(
    "--loss_funct",
    default="reward",
    type=str,
    help="loss function, either reward or mse",
)

# training parameters
parser.add_argument(
    "--cuda", default=False, type=boolean_string, help="run model on GPU"
)
parser.add_argument(
    "--n_runs", default=50, type=int, help="number of independent training runs"
)
parser.add_argument(
    "--n_episodes", default=200, type=int, help="number of training episodes"
)
parser.add_argument(
    "--perform_sgd",
    default=True,
    type=boolean_string,
    help="turn supervised update on/off",
)
parser.add_argument(
    "--perform_hebb",
    default=True,
    type=boolean_string,
    help="turn hebbian update on/off",
)
parser.add_argument(
    "--training_schedule",
    type=str,
    default="blocked",
    help="either interleaved or blocked",
)
parser.add_argument(
    "--log-interval", default=50, type=int, help="log very n training steps"
)
parser.add_argument(
    "--ctx_weights",
    default=False,
    type=boolean_string,
    help="scale context weights (yes/no)",
)

# io params
parser.add_argument(
    "--verbose",
    default=True,
    type=boolean_string,
    help="verbose mode, print all logs to stdout",
)
parser.add_argument(
    "--save_results",
    default=True,
    type=boolean_string,
    help="save model and results (yes/no)",
)
parser.add_argument("--save_dir", default="simu1", help="save dir for model outputs")


# tuner params
parser.add_argument(
    "--hpo_fixedseed",
    default=False,
    type=boolean_string,
    help="fixed random seed for hpo",
)

parser.add_argument(
    "--hpo_scheduler",
    default=None,
    type=str,
    help="trial scheduler, currently supports [asha, bohb, None]",
)

parser.add_argument(
    "--hpo_searcher",
    default=None,
    type=str,
    help="search algo, currently supports [bohb, None]",
)


# miscellaneous
parser.add_argument("--seed", default=1234, type=int, help="random seed")
