'''
find amount of sluggishness needed to reproduce human choice patterns

'''
import torch 
import pickle
from pathlib import Path 
import datetime 
import numpy as np
from utils.data import make_dataset
from utils.nnet import get_device
from scipy.io import loadmat
from logger import MetricLogger
from model import ChoiceNet
from trainer import Optimiser, train_model
from parameters import parser
from joblib import Parallel, delayed
from scipy.stats import spearmanr

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()

# load choicemats 
choicemats = loadmat('datasets/human_choices.mat')
cmat_a = choicemats['cmat_b_north'].mean(0)
cmat_b = choicemats['cmat_b_south'].mean(0)
choices_blocked = np.hstack((cmat_a.flatten(),cmat_b.flatten()))    
cmat_a = choicemats['cmat_i_north'].mean(0)
cmat_b = choicemats['cmat_i_south'].mean(0)
choices_interleaved = np.hstack((cmat_a.flatten(),cmat_b.flatten()))    


def execute_run(sv):
    # set amount of sluggishness
    args.ctx_avg_window = sv
    # get (cuda) device
    args.device,_ = get_device(args.cuda)
    
    # get dataset
    dataset = make_dataset(args)
    
    # instantiate logger, model and optimiser
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger = MetricLogger(save_dir)
    model = ChoiceNet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_model(args, model,optim,dataset, logger)

    # compute losses 
    y_out = np.array(logger.all_y_out).squeeze()[1,:]
    loss_blocked = np.mean(np.power(choices_blocked-y_out,2))
    loss_interleaved = np.mean(np.power(choices_interleaved-y_out,2))

    # compute correlations between human and model choices 
    X_blocked = np.stack((choices_blocked,y_out))
    X_interleaved = np.stack((choices_interleaved,y_out))
    # corrs = [np.corrcoef(X_blocked)[0,1],np.corrcoef(X_interleaved)[0,1]]
    corrs = [spearmanr(X_blocked[0,:],X_blocked[1,:])[0],spearmanr(X_interleaved[0,:],X_interleaved[1,:])[0]]
    ## return losses
    losses = [loss_blocked,loss_interleaved]
    return losses + corrs + [y_out]

    


if __name__ == "__main__":

    # overwrite standard parameters
    args.cuda = False
    args.ctx_scaling = 2
    args.lrate_sgd=0.03
    args.lrate_hebb=0.03
    args.weight_init=1e-3
    args.save_results = False 
    args.gating ='SLA'
    args.centering = 'True'
    args.verbose = False
    args.ctx_avg = True
    args.training_schedule='interleaved'
    args.loss_funct = 'rew_on_sigmoid'


    sluggish_vals = np.arange(1,401,1)
    results = Parallel(n_jobs=32,verbose=10)(delayed(execute_run)(sv) for sv in sluggish_vals)
    
    with open('fit_sluggishness_results.pkl','wb') as f:
        pickle.dump(results,f)
   