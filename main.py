import argparse 
import torch 

from utils.data import make_dataset
from utils.nnet import get_device, from_gpu

from model import Nnet
from trainer import Optimiser, train_model


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# parameters 
parser = argparse.ArgumentParser(description='CL simulations')

# data parameters 
parser.add_argument('--ctx_scaling',default=2, type=int,help='scaling of context signal')
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

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


if __name__ == "__main__":
    
    # get (cuda) device
    args.device,_ = get_device(args.cuda)
    
    # get dataset
    dataset = make_dataset(args)
    
    

    # instantiate model and optimiser
    model = Nnet(args).to(args.device)
    optim = Optimiser(args)

    # test: create test sets 
    x_a = torch.from_numpy(dataset['x_task_a']).float().to(args.device)
    r_a = torch.from_numpy(dataset['y_task_a']).float().to(args.device)

    x_b = torch.from_numpy(dataset['x_task_b']).float().to(args.device)
    r_b = torch.from_numpy(dataset['y_task_b']).float().to(args.device)
    
    y_a = model(x_a)
    y_b = model(x_b)

    # test: loss 
    loss_a = optim.loss_funct(r_a,y_a)
    loss_b = optim.loss_funct(r_b,y_b)
         
    print('loss at init: task a: {:.4f}, task b {:.4f}'.format(from_gpu(loss_a).ravel()[0],from_gpu(loss_b).ravel()[0]))
    # train model
    train_model(args, model,optim,dataset)
    y_a = model(x_a)
    y_b = model(x_b)
    loss_a = optim.loss_funct(r_a,y_a)
    loss_b = optim.loss_funct(r_b,y_b)
         
    print('loss after training: task a: {:.4f}, task b {:.4f}'.format(from_gpu(loss_a).ravel()[0],from_gpu(loss_b).ravel()[0]))
    
       

    # evaluate model 
    print('todo model eval')

    # save results 
    print('todo save results')