import argparse 


def boolean_string(s):
    '''
    helper function, turns string into boolean variable    
    '''
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



# parameters 
parser = argparse.ArgumentParser(description='Hebbian Continual Learning simulations')

# data parameters 
parser.add_argument('--ctx_scaling',default=2, type=float,help='scaling of context signal')
parser.add_argument('--ctx_avg',default=True, type=boolean_string, help='context averaging')
parser.add_argument('--ctx_avg_window',default=50,type=int, help='ctx avg window width')
parser.add_argument('--centering',default='True', type=boolean_string,help='centering of data')

# network parameters 
parser.add_argument('--n_features',default=27, type=int, help='number of stimulus units')
parser.add_argument('--n_out',default=1, type=int, help='number of output units')
parser.add_argument('--n_hidden',default=100, type=int, help='number of hidden units')
parser.add_argument('--weight_init',default=1e-5,type=float,help='initial weight scale')

# optimiser parameters
parser.add_argument('--lrate_sgd', default=1e-2,type=float, help='learning rate for SGD')
parser.add_argument('--lrate_hebb', default=0.01,type=float, help='learning rate for hebbian update')
parser.add_argument('--hebb_normaliser', default=10.0,type=float, help='normalising const. for hebbian update')
parser.add_argument('--gating',default='SLA',help='any of: None, manual, GHA, SLA')
parser.add_argument('--loss_funct',default='reward',type=str,help='loss function, either reward or mse')

# training parameters 
parser.add_argument('--cuda', default=True, type=boolean_string, help='run model on GPU')
parser.add_argument('--n_runs', default=10, type=int, help='number of independent training runs')
parser.add_argument('--n_episodes', default=200, type=int, help='number of training episodes')
parser.add_argument('--perform_sgd',default=True, type=boolean_string, help='turn supervised update on/off')
parser.add_argument('--perform_hebb',default=True, type=boolean_string, help='turn hebbian update on/off')
parser.add_argument('--training_schedule',default='blocked',help='either interleaved or blocked')
parser.add_argument('--log-interval',default=100,type=int,help='log very n training steps')

# debug params
parser.add_argument('--verbose',default=True, type=boolean_string, help='verbose mode, print all logs to stdout')
parser.add_argument('--save_results',default=False,type=boolean_string,help='save model and results (yes/no)')