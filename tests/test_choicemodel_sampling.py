import os
import sys
import numpy as np

root_path = os.path.realpath("./")
sys.path.append(root_path)

import pytest  # noqa E402
from hebbcl.parameters import parser  # noqa E402
from utils import eval, choicemodel  # noqa E402


def parse_args(args):
    return parser.parse_args(args)


class TestSampling:
    def test_acc_factorised(self):
        """tests whether sampling from factorised model yields 100% accuracy"""
        _, _, cmats = eval.gen_behav_models()
        cmat_fact_a = cmats[0, 0, :, :]
        cmat_fact_b = cmats[0, 1, :, :]
        acc_fact = choicemodel.compute_sampled_accuracy(cmat_fact_a, cmat_fact_b)
        assert np.round(acc_fact, 2) == 1.00

    def test_acc_linear(self):
        """tests whether sampling from linear model yields 80% correct"""
        _, _, cmats = eval.gen_behav_models()

        cmat_lin_a = cmats[1, 0, :, :]
        cmat_lin_b = cmats[1, 1, :, :]
        acc_lin = choicemodel.compute_sampled_accuracy(cmat_lin_a, cmat_lin_b)
        assert np.round(acc_lin, 2) == 0.80


if __name__ == "__main__":
    pytest.main([__file__])
