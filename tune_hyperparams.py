import torch
import pickle
import ray

ray.init()

from ray import tune

from utils.nnet import get_device
from hebbcl.parameters import parser
from hebbcl.tuner import run_raytune


# parse arguments
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# actually, let's not use a gpu for now
args.cuda = False

# get (cuda) device
args.device, _ = get_device(args.cuda)



if __name__ == "__main__":

    # configuration for ray tune
    if args.gating == "SLA":
        config = {
            "lrate_sgd": tune.loguniform(1e-4, 1e-1),
            "lrate_hebb": tune.loguniform(1e-4, 1e-1),
            "normaliser": tune.uniform(1, 20),
            "n_episodes": tune.choice([200, 500, 1000]),
            "sla": tune.grid_search([0, 1]),
        }

        results = run_raytune(config, args)

        with open("raytune_results_sla.pickle", "wb") as f:
            pickle.dump(results, f)

    elif args.gating == "manual":
        args.cuda = False
        args.centering = False
        args.ctx_scaling = 1

        config = {
            "lrate_sgd": tune.loguniform(1e-3, 1e-1),
            "n_episodes": tune.choice([200, 500, 1000]),
            "weight_init": tune.loguniform(1e-4, 1e-3),
        }

        results = run_raytune(config, args)

        with open("raytune_results_manualgating.pickle", "wb") as f:
            pickle.dump(results, f)
