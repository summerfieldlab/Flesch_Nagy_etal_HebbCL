'''
collects multiple runs per simulation
'''
import torch 
from pathlib import Path 
import datetime 

from utils.data import make_dataset
from utils.nnet import get_device

from logger import MetricLogger
from model import Gatednet, Nnet
from trainer import Optimiser, train_model
from parameters import parser
from joblib import Parallel, delayed

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()

def train_model(i_run):
    print('run {} / {}'.format(str(i_run),str(args.n_runs)))
        
    # create checkpoint dir 
    run_name = 'run_'+str(i_run)
    save_dir = Path("checkpoints") / args.save_dir / run_name
    

    # get (cuda) device
    args.device,_ = get_device(args.cuda)
    
    # get dataset
    dataset = make_dataset(args)
    
    # instantiate logger, model and optimiser
    logger = MetricLogger(save_dir)
    if args.gating=='manual':
        model = Gatednet(args)
    else:
        model = Nnet(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_model(args, model,optim,dataset, logger)
    

    # save results 
    if args.save_results:
        save_dir.mkdir(parents=True)
        logger.save(model)

if __name__ == "__main__":
    
    Parallel(n_jobs=30,verbose=10)(delayed(train_model)(i_run) for i_run in range(args.n_runs))
      