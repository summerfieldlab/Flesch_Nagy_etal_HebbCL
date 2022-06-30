import torch
from pathlib import Path

import numpy as np
from utils.data import make_blobs_dataset
from utils.nnet import get_device

from hebbcl.logger import MetricLogger
from hebbcl.model import Nnet
from hebbcl.trainer import Optimiser, train_on_blobs
from hebbcl.parameters import parser
from joblib import Parallel, delayed

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


def execute_run(i_run):
    print("run {} / {}".format(str(i_run), str(args.n_runs)))

    # create checkpoint dir
    run_name = "run_" + str(i_run)
    save_dir = Path("checkpoints") / args.save_dir / run_name

    # get (cuda) device
    args.device, _ = get_device(args.cuda)

    # get dataset
    dataset = make_blobs_dataset(args)

    # instantiate logger, model and optimiser
    logger = MetricLogger(save_dir)
    model = Nnet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_on_blobs(args, model, optim, dataset, logger)

    # save results
    if args.save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.save(model)


if __name__ == "__main__":

    # BASELINE NETWORK -------------------------------------------------
    args.cuda = False
    args.n_episodes = 8
    args.ctx_scaling = 5
    args.lrate_sgd = 0.2
    args.lrate_hebb = 0.0093
    args.weight_init = 1e-2
    args.save_results = True
    args.gating = "None"
    args.perform_hebb = False
    args.centering = True
    args.verbose = False
    args.ctx_avg = True
    args.ctx_avg_type = "ema"
    args.training_schedule = "interleaved"
    args.n_runs = 50

    sluggish_vals = np.linspace(0.05, 1, 30)
    for ii, sv in enumerate(sluggish_vals):
        args.ctx_avg_alpha = sv
        args.save_dir = "sluggish_baseline_int_8episodes_sv" + str(ii)
        Parallel(n_jobs=-1, verbose=10)(
            delayed(execute_run)(i_run) for i_run in range(args.n_runs)
        )

    # OJA NETWORK BLOCKED ---------------------------------------------
    # overwrite standard parameters
    args.cuda = False
    args.n_episodes = 8
    args.ctx_scaling = 3
    args.lrate_sgd = 0.09207067771676251
    args.lrate_hebb = 0.0039883754510576805
    args.weight_init = 1e-2
    args.save_results = True
    args.gating = "oja_ctx"
    args.centering = True
    args.verbose = False
    args.ctx_avg = True
    args.ctx_avg_type = "ema"
    args.training_schedule = "blocked"
    args.n_runs = 50

    sluggish_vals = np.linspace(0.05, 1, 30)
    for ii, sv in enumerate(sluggish_vals):
        args.ctx_avg_alpha = sv
        args.save_dir = "sluggish_oja_blocked_8episodes_sv" + str(ii)
        Parallel(n_jobs=-1, verbose=10)(
            delayed(execute_run)(i_run) for i_run in range(args.n_runs)
        )

    # OJA NETWORK INTERLEAVED -----------------------------------------
    # overwrite standard parameters
    args.cuda = False
    args.n_episodes = 8
    args.ctx_scaling = 3
    args.lrate_sgd = 0.08710014100174149
    args.lrate_hebb = 0.005814333717889643
    args.weight_init = 1e-2
    args.save_results = True
    args.gating = "oja_ctx"
    args.centering = True
    args.verbose = False
    args.ctx_avg = True
    args.ctx_avg_type = "ema"
    args.training_schedule = "interleaved"
    args.n_runs = 50

    sluggish_vals = np.linspace(0.05, 1, 30)
    for ii, sv in enumerate(sluggish_vals):
        args.ctx_avg_alpha = sv
        args.save_dir = "sluggish_oja_int_8episodes_sv" + str(ii)
        Parallel(n_jobs=-1, verbose=10)(
            delayed(execute_run)(i_run) for i_run in range(args.n_runs)
        )
