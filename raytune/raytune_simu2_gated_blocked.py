'''
hyperparameter tuner for simulation2: gated network with blocked data ("reward" loss)
'''
import torch 
import numpy as np
import pandas as pd
import os 
import pickle 
import ray
ray.init()

from ray import tune 

from utils.data import make_dataset
from utils.nnet import get_device, from_gpu
from utils.eval import compute_accuracy

from model import Nnet, Gatednet
from trainer import Optimiser
from parameters import parser 


# parse arguments 
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

# actually, let's not use a gpu for now
args.cuda=False

# get (cuda) device
args.device,_ = get_device(args.cuda)



def train_nnet_with_ray(config):
    '''
    trains neural network model
    '''

    # get dataset
    data = make_dataset(args)

    # replace args with ray tune config   

    args.lrate_sgd = config['lrate_sgd']
    args.n_episodes = config['n_episodes']        
    # args.weight_init = config['weight_init']

    # instantiate model and optimiser 
    model = Gatednet(args)
    optim = Optimiser(args) 

    # send model to GPU (optional)
    model = model.to(args.device)

    # send data to gpu (optional)
    x_train = torch.from_numpy(data['x_train']).float().to(args.device)
    y_train = torch.from_numpy(data['y_train']).float().to(args.device)

    x_both = torch.from_numpy(np.concatenate((data['x_task_a'],data['x_task_b']),axis=0)).float().to(args.device)
    r_both = torch.from_numpy(np.concatenate((data['y_task_a'],data['y_task_b']),axis=0)).float().to(args.device)

    # loop over data and apply optimiser
    idces = np.arange(len(x_train))
    for ii, x,y in zip(idces,x_train,y_train):
        optim.step(model,x,y)
        
        if ii%args.log_interval==0:
            # obtain loss/acc metrics
            loss_both = from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0]         
            acc_both = compute_accuracy(r_both,model(x_both))            
            # send metrics to ray tune 
            tune.report(mean_loss=loss_both,mean_acc=acc_both)



if __name__ == "__main__":

    # configuration for ray tune 
    args.gating = 'manual'
    args.cuda = False 
    args.centering = False
    args.ctx_scaling = 3
    args.ctx_avg = False
    args.training_schedule= 'blocked'
    args.loss_funct = 'reward'
    config = {
        'lrate_sgd':tune.loguniform(1e-4,1e-1),
        'n_episodes':tune.choice([200,500,1000])        
    }

    # run ray tune
    analysis = tune.run(train_nnet_with_ray,
                        config=config,
                        num_samples=1000,
                        metric="mean_acc", 
                        mode="max",
                        resources_per_trial={"cpu": 1, "gpu": 0})
    # get best configuration
    best_cfg = analysis.get_best_config(metric="mean_acc", mode="max") 
    print("Best config: ", best_cfg)

    # results as dataframe 
    df = analysis.results_df
    results = {
        'df':df,
        'best':best_cfg
    }
    # save results to disk
    with open('results_raytune_simu2_gated_blocked_rew.pkl','wb') as f:
        pickle.dump(results,f)


    
