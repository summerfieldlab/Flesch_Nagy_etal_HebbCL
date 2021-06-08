import torch, time, datetime, pickle
import numpy as np

from utils.nnet import from_gpu
from utils.eval import *



class MetricLogger():
    def __init__(self,save_dir):
        self.save_log = save_dir 
        
        # performance metrics: 
        self.losses_total = []
        self.losses_1st = []
        self.losses_2nd = []
        self.acc_total = []
        self.acc_1st = []
        self.acc_2nd = []
        
        # layer-wise activity patterns :
        self.all_x_hidden = []
        self.all_y_hidden = []
        self.all_y_out = []

        # relative weight change:
        self.w_h0 = []
        self.w_y0 = []
        self.w_relchange_hxs = []
        self.w_relchange_yh = []

        # task-specificity of units:
        self.n_dead = []
        self.n_local = []
        self.n_only_a = []
        self.n_only_b = []
        self.hidden_dotprod = []

        # misc:
        self.record_time = time.time()
        

    def log_init(self, model):
        '''
        log initial values (such as weights after random init)

        '''
        self.w_h0 = from_gpu(model.W_h)
        self.w_y0 = from_gpu(model.W_o)
        
    

    def log_step(self, model,optim,x_a,x_b,x_both,r_a,r_b,r_both):
        '''
        log a single training step  
        '''
        # accuracy/ loss 
        self.losses_total.append(from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0])
        self.losses_1st.append(from_gpu(optim.loss_funct(r_a, model(x_a))).ravel()[0])
        self.losses_2nd.append(from_gpu(optim.loss_funct(r_b, model(x_b))).ravel()[0])
        self.acc_total.append(compute_accuracy(r_both,model(x_both)))
        self.acc_1st.append(compute_accuracy(r_a,model(x_a)))
        self.acc_2nd.append(compute_accuracy(r_b,model(x_b)))

        # weight change
        self.w_relchange_hxs.append(compute_relchange(self.w_h0,from_gpu(model.W_h)))
        self.w_relchange_yh.append(compute_relchange(self.w_y0,from_gpu(model.W_o)))

        # sparsity        
        model.forward(x_both)
        n_dead,n_local,n_a,n_b,dotprod = compute_sparsity_stats(from_gpu(model.y_h).T)
        self.n_dead.append(n_dead)
        self.n_local.append(n_local)
        self.n_only_a.append(n_a) 
        self.n_only_b.append(n_b)
        self.hidden_dotprod.append(dotprod)   
        

    
    def log_patterns(self,model,x_both):
        '''
        log patterns elicited in the network's various layers
        '''
        
        # (hidden) layer patterns 
        model.forward(x_both)
        self.all_x_hidden.append(from_gpu(model.x_h))
        self.all_y_hidden.append(from_gpu(model.y_h))
        self.all_y_out.append(from_gpu(model.y))

    def save(self,model):
        '''
        saves logs (and model) to disk
        '''
        results = {}
        results['losses_total'] = np.asarray(self.losses_total)
        results['losses_1st'] = np.asarray(self.losses_1st)
        results['losses_2nd'] = np.asarray(self.losses_2nd)
        results['acc_total'] = np.asarray(self.acc_total)
        results['acc_1st'] = np.asarray(self.acc_1st)
        results['acc_2nd'] = np.asarray(self.acc_2nd)

        results['all_x_hidden'] = np.asarray(self.all_x_hidden)
        results['all_y_hidden'] = np.asarray(self.all_y_hidden)
        results['all_y_out'] = np.asarray(self.all_y_out)

        results['w_relchange_hxs'] = np.asarray(self.w_relchange_hxs)
        results['w_relchange_yh'] = np.asarray(self.w_relchange_yh)

        results['n_dead'] = np.asarray(self.n_dead)
        results['n_local'] = np.asarray(self.n_local)
        results['n_only_a'] = np.asarray(self.n_only_a)
        results['n_only_b'] = np.asarray(self.n_only_b)
        results['hidden_dotprod'] = np.asarray(self.hidden_dotprod)

        # set filenames 
        fname_results = 'results.pickle'
        fname_model = 'model.pickle'

        # save results and model
        with open(self.save_log / fname_results,'wb') as f:
            pickle.dump(results,f)
        
        with open(self.save_log /fname_model,'wb') as f:
            pickle.dump(model,f)
        

