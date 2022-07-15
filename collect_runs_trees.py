import torch
from pathlib import Path


from utils.data import make_trees_dataset
from utils.nnet import get_device

from hebbcl.logger import LoggerFactory
from hebbcl.model import ModelFactory
from hebbcl.trainer import Optimiser, train_on_trees
from hebbcl.parameters import parser
from joblib import Parallel, delayed

args = parser.parse_args()
# overwrite cuda argument depending on GPU availability
args.cuda = args.cuda and torch.cuda.is_available()


def execute_run_trees(i_run):
    print("run {} / {}".format(str(i_run), str(args.n_runs)))

    # create checkpoint dir
    run_name = "run_" + str(i_run)
    save_dir = Path("checkpoints") / args.save_dir / run_name

    # get (cuda) device
    args.device, _ = get_device(args.cuda)

    # trees settings
    args.n_episodes = 100
    args.n_layers = 2
    args.n_hidden = 100
    args.n_features = 974

    # get dataset
    dataset = make_trees_dataset(args, filepath="./datasets/")

    # instantiate logger, model and optimiser
    logger = LoggerFactory.create(args, save_dir)
    model = ModelFactory.create(args)
    optim = Optimiser(args)

    # send model to GPU
    model = model.to(args.device)

    # train model
    train_on_trees(args, model, optim, dataset, logger)

    # save results
    if args.save_results:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.save(model)


if __name__ == "__main__":

    Parallel(n_jobs=-1, verbose=10)(
        delayed(execute_run_trees)(i_run) for i_run in range(args.n_runs)
    )
