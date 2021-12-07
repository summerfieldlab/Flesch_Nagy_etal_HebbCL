import numpy as np
import torch 

from utils.nnet import from_gpu


def compute_accuracy(y,y_):
    '''
    accuracy for this experiment is defined as matching signs (>= and < 0) for outputs and targets
    The category boundary trials are neglected.
    '''
    valid_targets = y!=0
    outputs = y_ > 0
    targets = y > 0
    return from_gpu(torch.mean((outputs[valid_targets]==targets[valid_targets]).float())).ravel()[0]
    


def compute_sparsity_stats(yout):
    '''
    computes task-specificity of activity patterns
    '''
    # yout is n_units x n_trials 
    # 1. average within contexts
    assert yout.shape==(100,50)
    x = np.vstack((np.mean(yout[:,0:25],1).T,np.mean(yout[:,25:-1],1).T))
    # should yield a 2xn_hidden vector
    # now count n dead units (i.e. return 0 in both tasks)
    n_dead = np.sum(~np.any(x,axis=0))
    # now count number of local units in total (only active in one task)
    n_local = np.sum(~(np.all(x,axis=0)) & np.any(x,axis=0))
    # count units only active in task a 
    n_only_A = np.sum(np.all(np.vstack((x[0,:]>0,x[1,:]==0)),axis=0))
    # count units only active in task b 
    n_only_B = np.sum(np.all(np.vstack((x[0,:]==0,x[1,:]>0)),axis=0))
    # compute dot product of hiden layer activations 
    h_dotprod = np.dot(x[0,:],x[1,:].T)
    # return all
    return n_dead, n_local, n_only_A, n_only_B, h_dotprod



def mse(y_,y):
    '''
    computes mean squared error between targets and outputs
    '''
    return .5*np.linalg.norm(y_-y,2)**2


def compute_relchange(w0,wt):    
    '''
    computes relative change of norm of weights 
    inputs:
    - w0: weights at initialisation
    - wt: weights at time t.
    output: 
    - (norm(wt)-norm(w0))/norm(w0)
    '''
    return (np.linalg.norm(wt.flatten())-np.linalg.norm(w0.flatten()))/np.linalg.norm(w0.flatten())
