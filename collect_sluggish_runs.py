'''
collects multiple runs for a range of sluggishness values
'''
import torch
from pathlib import Path

import numpy as np
from utils.data import make_dataset
from utils.nnet import get_device

from logger import MetricLogger
from model import Nnet
from trainer import Optimiser, train_model
from parameters import parser
from joblib import Parallel, delayed

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


def execute_run(i_run):
    print('run {} / {}'.format(str(i_run), str(args.n_runs)))

    # create checkpoint dir
    run_name = 'run_'+str(i_run)
    save_dir = Path("checkpoints") / args.save_dir / run_name

    # get (cuda) device
    args.device, _ = get_device(args.cuda)

    # get dataset
    dataset = make_dataset(args)

    # instantiate logger, model and optimiser
    logger = MetricLogger(save_dir)
    model = Nnet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_model(args, model, optim, dataset, logger)

    # save results
    if args.save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.save(model)


if __name__ == "__main__":

    # BASELINE NETWORK -------------------------------------------------
    args.cuda = False
    args.ctx_scaling = 5
    args.lrate_sgd = 0.2
    args.lrate_hebb = 0.0093
    args.weight_init = 1e-2
    args.save_results = True
    args.gating = 'None'
    args.centering = 'True'
    args.verbose = False
    args.ctx_avg = True
    args.ctx_avg_type = 'ema'
    args.training_schedule = 'interleaved'
    args.n_runs = 50
    # args.loss_funct = 'rew_on_sigmoid'

    sluggish_vals = np.linspace(0.05, 1, 20)
    for ii, sv in enumerate(sluggish_vals):
        args.ctx_avg_alpha = sv
        args.save_dir = 'sluggish_baseline_int_sv' + str(ii)
        Parallel(n_jobs=6, verbose=10)(delayed(execute_run)(i_run)
                                       for i_run in range(args.n_runs))

    # SLA NETWORK ------------------------------------------------------
    # overwrite standard parameters
    args.cuda = False
    args.ctx_scaling = 5
    args.lrate_sgd = 0.03
    args.lrate_hebb = 0.009
    args.weight_init = 1e-2
    args.save_results = True
    args.gating = 'SLA'
    args.centering = 'True'
    args.verbose = False
    args.ctx_avg = True
    args.ctx_avg_type = 'ema'
    args.training_schedule = 'interleaved'
    args.n_runs = 50
    # args.loss_funct = 'rew_on_sigmoid'

    sluggish_vals = np.linspace(0.05, 1, 20)
    for ii, sv in enumerate(sluggish_vals):
        args.ctx_avg_alpha = sv
        args.save_dir = 'sluggish_sla_int_sv' + str(ii)
        Parallel(n_jobs=6, verbose=10)(delayed(execute_run)(i_run)
                                       for i_run in range(args.n_runs))
