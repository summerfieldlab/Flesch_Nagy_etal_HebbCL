import torch 
from pathlib import Path 
import datetime 

from utils.data import make_dataset
from utils.nnet import get_device

from logger import MetricLogger
from model import Nnet
from trainer import Optimiser, train_model
from parameters import parser


args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


if __name__ == "__main__":
    
    # create checkpoint dir 
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    

    # get (cuda) device
    args.device,_ = get_device(args.cuda)
    
    # get dataset
    dataset = make_dataset(args)
    
    # instantiate logger, model and optimiser
    logger = MetricLogger(save_dir)
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