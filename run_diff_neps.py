import torch
from pathlib import Path

import numpy as np
from utils.data import make_blobs_dataset
from utils.nnet import get_device

from hebbcl.logger import MetricLogger1Hidden
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
    logger = MetricLogger1Hidden(save_dir)
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

    # REVISION: OJA NETWORK BLOCKED ---------------------------------------------
    # overwrite standard parameters
    args.cuda = False
    args.n_episodes = 200
    args.ctx_scaling = 2
    args.lrate_sgd = 0.03775369549108046
    args.lrate_hebb = 0.00021666673995458582
    args.weight_init = 1e-2
    args.save_results = True
    args.perform_hebb = True
    args.gating = "oja"
    args.centering = True
    args.verbose = False
    args.ctx_avg = False
    args.ctx_avg_type = "ema"
    args.training_schedule = "blocked"
    args.n_runs = 50

    n_eps = np.arange(200, 510, 25)
    for ii, ep in enumerate(n_eps):
        if ep != 200:
            args.n_episodes = ep
            args.save_dir = f"blobs_revision_{ep}episodes_blocked_oja"
            Parallel(n_jobs=6, verbose=10)(
                delayed(execute_run)(i_run) for i_run in range(args.n_runs)
            )
