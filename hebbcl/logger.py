import time
from typing import Tuple
import torch
import pickle
import pathlib
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
from hebbcl.trainer import Optimiser
from copy import deepcopy

from utils.nnet import from_gpu
from utils.eval import (
    compute_accuracy,
    compute_relchange,
    compute_sparsity_stats,
    make_dmat,
)


class LoggerFactory:
    @staticmethod
    def create(args, save_dir):
        if args.n_layers == 1:
            return MetricLogger1Hidden(save_dir)
        elif args.n_layers == 2:
            return MetricLogger2Hidden(save_dir)


class MetricLogger1Hidden:
    """logs experiment results"""

    def __init__(self, save_dir: pathlib.Path):
        """Constructor for metric logger

        Args:
            save_dir (pathlib.Path): path to save directory
        """
        self.save_log = save_dir

        self.scale_noise = [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.5,
            2,
        ]
        self.results = {}
        # performance metrics:
        self.results["losses_total"] = []
        self.results["losses_1st"] = []
        self.results["losses_2nd"] = []
        self.results["acc_total"] = []
        self.results["acc_1st"] = []
        self.results["acc_2nd"] = []
        self.results["acc_1st_noise"] = []
        self.results["acc_2nd_noise"] = []

        # layer-wise activity patterns :
        self.results["all_x_hidden"] = []
        self.results["all_y_hidden"] = []
        self.results["all_y_out"] = []
        self.results["task_a_sel"] = []
        self.results["task_b_sel"] = []

        # relative weight change amd context corrs:
        self.results["w_h0"] = []
        self.results["w_y0"] = []
        self.results["w_relchange_hxs"] = []
        self.results["w_relchange_yh"] = []
        self.results["w_context_corr"] = []

        # task-specificity of units:
        self.results["n_dead"] = []
        self.results["n_local"] = []
        self.results["n_only_a"] = []
        self.results["n_only_b"] = []
        self.results["hidden_dotprod"] = []

        # task-specificy via regression
        self.results["n_only_a_regr"] = []
        self.results["n_only_b_regr"] = []

        # misc:
        self.results["record_time"] = time.time()

    def log_init(self, model: torch.nn.Module):
        """log initial values (such as weights after random init)

        Args:
            model (torch.nn.Module): feedforward neural network
        """
        self.results["w_h0"] = deepcopy(from_gpu(model.W_h))
        self.results["w_y0"] = deepcopy(from_gpu(model.W_o))

    def log_step(
        self,
        model: torch.nn.Module,
        optim: Optimiser,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_both: torch.Tensor,
        r_a: torch.Tensor,
        r_b: torch.Tensor,
        r_both: torch.Tensor,
        f_both: torch.Tensor,
    ):
        """log a single training step

        Args:
            model (torch.nn.Module): feedfoward neural network
            optim (trainer.Optimiser): optimiser to perform SGD/Hebb
            x_a (torch.Tensor): test inputs task A
            x_b (torch.Tensor): test inputs task B
            x_both (torch.Tensor): test inputs both
            r_a (torch.Tensor): rewards, task A
            r_b (torch.Tensor): rewards task B
            r_both (torch.Tensor): rewards, both tasks
            f_both (torch.Tensor): feature values both tasks
        """
        # accuracy/ loss
        self.results["losses_total"].append(
            from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0]
        )
        self.results["losses_1st"].append(
            from_gpu(optim.loss_funct(r_a, model(x_a))).ravel()[0]
        )
        self.results["losses_2nd"].append(
            from_gpu(optim.loss_funct(r_b, model(x_b))).ravel()[0]
        )
        self.results["acc_total"].append(compute_accuracy(r_both, model(x_both)))
        self.results["acc_1st"].append(compute_accuracy(r_a, model(x_a)))
        self.results["acc_2nd"].append(compute_accuracy(r_b, model(x_b)))
        accs_noise_1st = []
        accs_noise_2nd = []

        for noiselvl in self.scale_noise:
            y_n = model(
                x_a
                + torch.from_numpy(
                    noiselvl * np.random.randn(len(x_a), x_a.shape[1])
                ).float()
            )
            accs_noise_1st.append(compute_accuracy(r_a, y_n))
            y_n = model(
                x_b
                + torch.from_numpy(
                    noiselvl * np.random.randn(len(x_b), x_a.shape[1])
                ).float()
            )
            accs_noise_2nd.append(compute_accuracy(r_b, y_n))

        self.results["acc_1st_noise"].append(accs_noise_1st)
        self.results["acc_2nd_noise"].append(accs_noise_2nd)

        # weight change and correlation
        self.results["w_relchange_hxs"].append(
            compute_relchange(self.results["w_h0"], from_gpu(model.W_h))
        )
        self.results["w_relchange_yh"].append(
            compute_relchange(self.results["w_y0"], from_gpu(model.W_o))
        )
        self.results["w_context_corr"].append(
            np.corrcoef(from_gpu(model.W_h)[-2:, :])[0, 1]
        )

        # sparsity
        model.forward(x_both[:50, :])
        n_dead, n_local, n_a, n_b, dotprod = compute_sparsity_stats(
            from_gpu(model.y_h).T
        )
        self.results["n_dead"].append(n_dead)
        self.results["n_local"].append(n_local)
        self.results["n_only_a"].append(n_a)
        self.results["n_only_b"].append(n_b)
        self.results["hidden_dotprod"].append(dotprod)

        if not hasattr(self, "dmat"):
            self.dmat = make_dmat(f_both[:50, :])
        yh = from_gpu(model.y_h)
        # assert yh.shape == (50, 100)
        n_ta, n_tb = self.check_selectivity(yh)
        self.results["n_only_a_regr"].append(n_ta)
        self.results["n_only_b_regr"].append(n_tb)

    def log_patterns(self, model: torch.nn.Module, x_both: torch.Tensor):
        """logs hidden layer activity patterns

        Args:
            model (torch.nn.Module): feedfoward neural network
            x_both (torch.Tensor): test inputs from both tasks
        """

        # (hidden) layer patterns
        model.forward(x_both)
        self.results["all_x_hidden"].append(from_gpu(model.x_h))
        self.results["all_y_hidden"].append(from_gpu(model.y_h))
        self.results["all_y_out"].append(from_gpu(model.y))

    def check_selectivity(self, yh: torch.Tensor) -> Tuple[np.int, np.int]:
        """performs linear regression to test single unit selectivity

        Args:
            yh (torch.Tensor): hidden layer outputs

        Returns:
            Tuple[np.int, np.int]: number of task a and task b selective units
        """
        selectivity_matrix = np.zeros((yh.shape[1], 6))
        for i_neuron in range(yh.shape[1]):
            y_neuron = yh[:, i_neuron]
            model = sm.OLS(zscore(y_neuron), self.dmat)
            regr_results = model.fit()
            # if only a single regressor is significant, store that neuron's selectivity
            if np.sum(regr_results.tvalues > 1.96) == 1:
                selectivity_matrix[
                    i_neuron,
                    np.where(regr_results.tvalues == np.max(regr_results.tvalues))[0][
                        0
                    ],
                ] = 1
        i_task_a = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 1)
            & (selectivity_matrix[:, 2] == 0)
            & (selectivity_matrix[:, 3] == 0)
        )
        i_task_b = (
            (selectivity_matrix[:, 0] == 0)
            & (selectivity_matrix[:, 1] == 0)
            & (selectivity_matrix[:, 2] == 1)
            & (selectivity_matrix[:, 3] == 0)
        )

        n_ta = np.sum(i_task_a)
        if n_ta > 0:
            self.results["task_a_sel"].append(yh[:, i_task_a].mean(1))
        else:
            self.results["task_a_sel"].append(np.zeros((50,)))

        n_tb = np.sum(i_task_b)
        if n_tb > 0:
            self.results["task_b_sel"].append(yh[:, i_task_b].mean(1))
        else:
            self.results["task_b_sel"].append(np.zeros((50,)))

        return n_ta, n_tb

    def save(
        self,
        model: torch.nn.Module,
        fname_results="results.pkl",
        fname_model="model.pkl",
    ):
        """saves the trained model and the logging results to disk

        Args:
            model (torch.nn.Module): the feedfoward neural network
            fname_results (str, optional): file name for results. Defaults to "results.pkl".
            fname_model (str, optional): file name for model. Defaults to "model.pkl".
        """

        for k, v in self.results.items():
            self.results[k] = np.asarray(v, dtype=object)
        # save results and model
        with open(self.save_log / fname_results, "wb") as f:
            pickle.dump(self.results, f)
            print(f"saved results to {self.save_log}")

        with open(self.save_log / fname_model, "wb") as f:
            pickle.dump(model, f)


class MetricLogger2Hidden(MetricLogger1Hidden):
    def __init__(self, save_dir: pathlib.Path):
        super().__init__(save_dir)

        # add vars for 2nd hidden layer
        self.results["all_x_hidden2"] = []
        self.results["all_y_hidden2"] = []
        self.results["w_relchange_hxs2"] = []
        self.results["w_relchange_yh2"] = []
        self.results["task_a_sel2"] = []
        self.results["task_b_sel2"] = []
        self.results["w_h0_2"] = []
        self.results["w_context_corr2"] = []

        self.results["n_dead2"] = []
        self.results["n_local2"] = []
        self.results["n_only_a2"] = []
        self.results["n_only_b2"] = []
        self.results["hidden_dotprod2"] = []

        self.results["n_only_a_regr2"] = []
        self.results["n_only_b_regr2"] = []

    def log_init(self, model: torch.nn.Module):
        """log initial values (such as weights after random init)

        Args:
            model (torch.nn.Module): feedforward neural network
        """
        self.results["w_h0"] = deepcopy(from_gpu(model.W_h1))
        self.results["w_h0_2"] = deepcopy(from_gpu(model.W_h2))
        self.results["w_y0"] = deepcopy(from_gpu(model.W_o))

    def log_step(
        self,
        model: torch.nn.Module,
        optim: Optimiser,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        x_both: torch.Tensor,
        x_pattern: torch.Tensor,
        f_pattern: torch.Tensor,
        r_a: torch.Tensor,
        r_b: torch.Tensor,
        r_both: torch.Tensor,        
    ):
        """log a single training step

        Args:
            model (torch.nn.Module): feedfoward neural network
            optim (trainer.Optimiser): optimiser to perform SGD/Hebb
            x_a (torch.Tensor): test inputs task A
            x_b (torch.Tensor): test inputs task B
            x_both (torch.Tensor): test inputs both 
            x_pattern (torch.Tensor) first 25 from task a and 25 from task b
            f_pattern (torch.Tensor) feature vals for above
            r_a (torch.Tensor): rewards, task A
            r_b (torch.Tensor): rewards task B
            r_both (torch.Tensor): rewards, both tasks            
        """
        # accuracy/ loss
        self.results["losses_total"].append(
            from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0]
        )
        self.results["losses_1st"].append(
            from_gpu(optim.loss_funct(r_a, model(x_a))).ravel()[0]
        )
        self.results["losses_2nd"].append(
            from_gpu(optim.loss_funct(r_b, model(x_b))).ravel()[0]
        )
        self.results["acc_total"].append(compute_accuracy(r_both, model(x_both)))
        self.results["acc_1st"].append(compute_accuracy(r_a, model(x_a)))
        self.results["acc_2nd"].append(compute_accuracy(r_b, model(x_b)))
        accs_noise_1st = []
        accs_noise_2nd = []

        for noiselvl in self.scale_noise:
            y_n = model(
                x_a
                + torch.from_numpy(
                    noiselvl * np.random.randn(len(x_a), x_a.shape[1])
                ).float()
            )
            accs_noise_1st.append(compute_accuracy(r_a, y_n))
            y_n = model(
                x_b
                + torch.from_numpy(
                    noiselvl * np.random.randn(len(x_b), x_a.shape[1])
                ).float()
            )
            accs_noise_2nd.append(compute_accuracy(r_b, y_n))

        self.results["acc_1st_noise"].append(accs_noise_1st)
        self.results["acc_2nd_noise"].append(accs_noise_2nd)

        # weight change and correlation
        self.results["w_relchange_hxs"].append(
            compute_relchange(self.results["w_h0"], from_gpu(model.W_h1))
        )
        self.results["w_relchange_hxs2"].append(
            compute_relchange(self.results["w_h0_2"], from_gpu(model.W_h2))
        )
        self.results["w_relchange_yh"].append(
            compute_relchange(self.results["w_y0"], from_gpu(model.W_o))
        )
        self.results["w_context_corr"].append(
            np.corrcoef(from_gpu(model.W_h1)[-2:, :])[0, 1]
        )
        self.results["w_context_corr2"].append(
            np.corrcoef(from_gpu(model.W_h2)[-2:, :])[0, 1]
        )

        # sparsity
        model.forward(x_pattern)
        n_dead, n_local, n_a, n_b, dotprod = compute_sparsity_stats(
            from_gpu(model.y_h1).T
        )
        self.results["n_dead"].append(n_dead)
        self.results["n_local"].append(n_local)
        self.results["n_only_a"].append(n_a)
        self.results["n_only_b"].append(n_b)
        self.results["hidden_dotprod"].append(dotprod)

        n_dead, n_local, n_a, n_b, dotprod = compute_sparsity_stats(
            from_gpu(model.y_h2).T
        )
        self.results["n_dead2"].append(n_dead)
        self.results["n_local2"].append(n_local)
        self.results["n_only_a2"].append(n_a)
        self.results["n_only_b2"].append(n_b)
        self.results["hidden_dotprod2"].append(dotprod)

        if not hasattr(self, "dmat"):
            self.dmat = make_dmat(f_pattern)

        yh = from_gpu(model.y_h1)
        # assert yh.shape == (50, 100)
        n_ta, n_tb = self.check_selectivity(yh)
        self.results["n_only_a_regr"].append(n_ta)
        self.results["n_only_b_regr"].append(n_tb)

        yh = from_gpu(model.y_h2)
        # assert yh.shape == (50, 100)
        n_ta, n_tb = self.check_selectivity(yh)
        self.results["n_only_a_regr2"].append(n_ta)
        self.results["n_only_b_regr2"].append(n_tb)

    def log_patterns(self, model: torch.nn.Module, x_pattern: torch.Tensor):
        """logs hidden layer activity patterns

        Args:
            model (torch.nn.Module): feedfoward neural network
            x_pattern (torch.Tensor): test inputs from both tasks (50 exemplars)
        """

        # (hidden) layer patterns
        model.forward(x_pattern)
        self.results["all_x_hidden"].append(from_gpu(model.x_h1))
        self.results["all_y_hidden"].append(from_gpu(model.y_h1))
        self.results["all_x_hidden2"].append(from_gpu(model.x_h2))
        self.results["all_y_hidden2"].append(from_gpu(model.y_h2))
        self.results["all_y_out"].append(from_gpu(model.y))
