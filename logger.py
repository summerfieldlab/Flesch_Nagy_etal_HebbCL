import time
from typing import Tuple
import torch
import pickle
import pathlib
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
import trainer

from utils.nnet import from_gpu
from utils.eval import *


class MetricLogger:
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
        # performance metrics:
        self.losses_total = []
        self.losses_1st = []
        self.losses_2nd = []
        self.acc_total = []
        self.acc_1st = []
        self.acc_2nd = []
        self.acc_1st_noise = []
        self.acc_2nd_noise = []

        # layer-wise activity patterns :
        self.all_x_hidden = []
        self.all_y_hidden = []
        self.all_y_out = []
        self.task_a_sel = []
        self.task_b_sel = []

        # relative weight change amd context corrs:
        self.w_h0 = []
        self.w_y0 = []
        self.w_relchange_hxs = []
        self.w_relchange_yh = []
        self.w_context_corr = []

        # task-specificity of units:
        self.n_dead = []
        self.n_local = []
        self.n_only_a = []
        self.n_only_b = []
        self.hidden_dotprod = []

        # task-specificy via regression
        self.n_only_a_regr = []
        self.n_only_b_regr = []

        # misc:
        self.record_time = time.time()

    def log_init(self, model: torch.nn.Module):
        """log initial values (such as weights after random init)

        Args:
            model (torch.nn.Module): feedforward neural network
        """
        self.w_h0 = from_gpu(model.W_h)
        self.w_y0 = from_gpu(model.W_o)

    def log_step(
        self,
        model: torch.nn.Module,
        optim: trainer.Optimiser,
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
        self.losses_total.append(
            from_gpu(optim.loss_funct(r_both, model(x_both))).ravel()[0]
        )
        self.losses_1st.append(from_gpu(optim.loss_funct(r_a, model(x_a))).ravel()[0])
        self.losses_2nd.append(from_gpu(optim.loss_funct(r_b, model(x_b))).ravel()[0])
        self.acc_total.append(compute_accuracy(r_both, model(x_both)))
        self.acc_1st.append(compute_accuracy(r_a, model(x_a)))
        self.acc_2nd.append(compute_accuracy(r_b, model(x_b)))
        accs_noise_1st = []
        accs_noise_2nd = []

        for noiselvl in self.scale_noise:
            y_n = model(
                x_a + torch.from_numpy(noiselvl * np.random.randn(25, 27)).float()
            )
            accs_noise_1st.append(compute_accuracy(r_a, y_n))
            y_n = model(
                x_b + torch.from_numpy(noiselvl * np.random.randn(25, 27)).float()
            )
            accs_noise_2nd.append(compute_accuracy(r_b, y_n))

        self.acc_1st_noise.append(accs_noise_1st)
        self.acc_2nd_noise.append(accs_noise_2nd)

        # weight change and correlation
        self.w_relchange_hxs.append(compute_relchange(self.w_h0, from_gpu(model.W_h)))
        self.w_relchange_yh.append(compute_relchange(self.w_y0, from_gpu(model.W_o)))
        self.w_context_corr.append(np.corrcoef(from_gpu(model.W_h)[-2:, :])[0, 1])

        # sparsity
        model.forward(x_both)
        n_dead, n_local, n_a, n_b, dotprod = compute_sparsity_stats(
            from_gpu(model.y_h).T
        )
        self.n_dead.append(n_dead)
        self.n_local.append(n_local)
        self.n_only_a.append(n_a)
        self.n_only_b.append(n_b)
        self.hidden_dotprod.append(dotprod)

        if not hasattr(self, "dmat"):
            self.dmat = make_dmat(f_both)
        yh = from_gpu(model.y_h)
        assert yh.shape == (50, 100)
        n_ta, n_tb = self.check_selectivity(yh)
        self.n_only_a_regr.append(n_ta)
        self.n_only_b_regr.append(n_tb)

    def log_patterns(self, model: torch.nn.Module, x_both: torch.Tensor):
        """logs hidden layer activity patterns

        Args:
            model (torch.nn.Module): feedfoward neural network
            x_both (torch.Tensor): test inputs from both tasks
        """

        # (hidden) layer patterns
        model.forward(x_both)
        self.all_x_hidden.append(from_gpu(model.x_h))
        self.all_y_hidden.append(from_gpu(model.y_h))
        self.all_y_out.append(from_gpu(model.y))

    def check_selectivity(self, yh: torch.Tensor) -> Tuple[np.int, np.int]:
        """performs linear regression to test single unit selectivity

        Args:
            yh (torch.Tensor): hidden layer outputs

        Returns:
            Tuple[np.int, np.int]: number of task a and task b selective units
        """
        selectivity_matrix = np.zeros((100, 6))
        for i_neuron in range(100):
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
            self.task_a_sel.append(yh[:, i_task_a].mean(1))
        else:
            self.task_a_sel.append(np.zeros((50,)))

        n_tb = np.sum(i_task_b)
        if n_tb > 0:
            self.task_b_sel.append(yh[:, i_task_b].mean(1))
        else:
            self.task_b_sel.append(np.zeros((50,)))

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
        results = {}
        results["losses_total"] = np.asarray(self.losses_total)
        results["losses_1st"] = np.asarray(self.losses_1st)
        results["losses_2nd"] = np.asarray(self.losses_2nd)
        results["acc_total"] = np.asarray(self.acc_total)
        results["acc_1st"] = np.asarray(self.acc_1st)
        results["acc_2nd"] = np.asarray(self.acc_2nd)
        results["acc_1st_noise"] = np.asarray(self.acc_1st_noise)
        results["acc_2nd_noise"] = np.asarray(self.acc_2nd_noise)

        results["all_x_hidden"] = np.asarray(self.all_x_hidden)
        results["all_y_hidden"] = np.asarray(self.all_y_hidden)
        results["all_y_out"] = np.asarray(self.all_y_out)

        results["w_relchange_hxs"] = np.asarray(self.w_relchange_hxs)
        results["w_relchange_yh"] = np.asarray(self.w_relchange_yh)
        results["w_context_corr"] = np.asarray(self.w_context_corr)
        results["n_dead"] = np.asarray(self.n_dead)
        results["n_local"] = np.asarray(self.n_local)
        results["n_only_a"] = np.asarray(self.n_only_a)
        results["n_only_b"] = np.asarray(self.n_only_b)
        results["n_only_a_regr"] = np.asarray(self.n_only_a_regr)
        results["n_only_b_regr"] = np.asarray(self.n_only_b_regr)
        results["hidden_dotprod"] = np.asarray(self.hidden_dotprod)
        results["task_a_sel"] = np.asarray(self.task_a_sel)
        results["task_b_sel"] = np.asarray(self.task_b_sel)

        # save results and model
        with open(self.save_log / fname_results, "wb") as f:
            pickle.dump(results, f)

        with open(self.save_log / fname_model, "wb") as f:
            pickle.dump(model, f)
