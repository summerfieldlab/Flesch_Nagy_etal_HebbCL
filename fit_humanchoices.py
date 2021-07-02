'''
runs sluggish SLA and finds sluggishness value that minimises difference between the model's learned outputs 
and choices observed in human participants
'''
import ray
ray.init()

from ray import tune 

import torch 
from pathlib import Path 
import datetime 
from scipy.io import loadmat
from utils.data import make_dataset
from utils.nnet import get_device

from logger import MetricLogger
from model import ChoiceNet
from trainer import Optimiser, train_model
from parameters import parser


args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


if __name__ == "__main__":

    # overwrite default parameters 
    args.save_results = False 
    args.gating ='SLA'
    args.centering = 'True'
    args.verbose = True
    args.ctx_avg = True
    args.ctx_avg_window = 1    
    args.training_schedule='interleaved'
    args.loss_funct = 'rew_on_sigmoid'

    
    # create checkpoint dir 
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    

    # get (cuda) device
    args.device,_ = get_device(args.cuda)
    
    # load choice matrices 
    choicemats = loadmat('datasets/human_choices.mat')
    # extract condition-specific cmats 
    if args.training_schedule=='blocked':
        cmat_a = choicemats['cmat_b_north'].mean(0)
        cmat_b = choicemats['cmat_b_south'].mean(0)

    elif args.training_schedule=='interleaved':
        cmat_a = choicemats['cmat_i_north'].mean(0)
        cmat_b = choicemats['cmat_i_south'].mean(0)
 
    # get dataset
    dataset = make_dataset(args)
    
    # instantiate logger, model and optimiser
    logger = MetricLogger(save_dir)
    model = ChoiceNet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_model(args, model,optim,dataset, logger)
       

    # save results 
    if args.save_results:
        save_dir.mkdir(parents=True)
        logger.save(model)