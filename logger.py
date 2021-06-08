import torch, time, datetime
import numpy as np

from utils.nnet import from_gpu
from utils.eval import compute_accuracy



class MetricLogger():
    def __init__(self,save_dir):
        self.save_log = save_dir / 'log'
        print('todo: metriclogger/init: define vars to log')

        self.record_time = time.time()

    def log_step(self):
        '''
        log a single training step  
        '''
        print('todo: metriclogger/log_step')
        pass 


    def save(self):
        '''
        saves logs to disk
        '''
        print('todo: metriclogger/saver')
        pass 


    def evaluate(self,model,dataset):
        '''
        evaluates model
        '''
        print('todo: metriclogger/evaluate')
        pass 