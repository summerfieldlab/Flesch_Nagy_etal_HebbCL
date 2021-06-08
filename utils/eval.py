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
    