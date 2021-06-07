import argparse 
import torch 

from utils.data import make_dataset
from utils.nnet import get_device, from_gpu

from model import Nnet
from trainer import Optimiser 

# parameters 
parser = argparse.ArgumentParser(description='CL simulations')

# data parameters 
parser.add_argument('--ctx_scaling',default=2, type=int,help='scaling of context signal')
parser.add_argument('--ctx_avg',default=True, help='context averaging')
parser.add_argument('--ctx_avg_window',default=50,type=int, help='ctx avg window width')
parser.add_argument('--centering',default='True',help='centering of data')

# network parameters 
parser.add_argument('--n_features',default=27, type=int, help='number of stimulus units')
parser.add_argument('--n_out',default=1, type=int, help='number of output units')
parser.add_argument('--n_hidden',default=100, type=int, help='number of hidden units')
parser.add_argument('--weight_init',default=1e-3,type=float,help='initial weight scale')

# optimiser parameters
parser.add_argument('--lrate_sgd', default=1e-3,type=float, help='learning rate for SGD')
parser.add_argument('--lrate_hebb', default=0.01,type=float, help='learning rate for hebbian update')
parser.add_argument('--hebb_normaliser', default=10.0,type=float, help='normalising const. for hebbian update')
parser.add_argument('--gating',default='SLA',help='any of: None, manual, GHA, SLA')

# training parameters 
parser.add_argument('--cuda', default=True, help='run model on GPU')
parser.add_argument('--n_runs', default=10, type=int, help='number of independent training runs')
parser.add_argument('--n_episodes', default=2000, type=int, help='number of training episodes')
parser.add_argument('--perform_sgd',default=True, help='turn supervised update on/off')
parser.add_argument('--perform_hebb',default=True, help='turn hebbian update on/off')
parser.add_argument('--training_schedule',default='interleaved',help='either interleaved or blocked')

# debug params
parser.add_argument('--verbose',default=True,help='verbose mode, print all logs to stdout')

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


if __name__ == "__main__":
    
    # get (cuda) device
    device,_ = get_device(args.cuda)
    
    # get dataset
    dataset = make_dataset(args)
    x_train = torch.from_numpy(dataset['x_task_a']).float().to(device)
    y_train = torch.from_numpy(dataset['y_task_a']).float().to(device)
    

    # instantiate model 
    model = Nnet(args).to(device)

    # test: forward pass 
    y_ = model(x_train)
    

    # test: backward pass 
    loss = model.loss_funct(y_train,y_)
    
         
    print('loss at init: {:.4f}'.format(from_gpu(loss).ravel()[0]))
    # instantiate optimiser 
    optim = Optimiser(args)
    for ii in range(10000):
        optim.sgd_update(model,x_train,y_train)
    y_ = model(x_train)
    loss = model.loss_funct(y_train,y_)
    print('loss after sgd: {:.4f}'.format(from_gpu(loss).ravel()[0]))
    
    
    # train model
    # trainer(args, model,optim)
    

    # evaluate model 
    print('todo model eval')

    # save results 
    print('todo save results')