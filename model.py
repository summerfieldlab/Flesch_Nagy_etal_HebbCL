import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse


class Nnet(nn.Module):
    """simple feed forward neural network with a single hidden layer"""

    def __init__(self, args: argparse.ArgumentParser):
        """constructor for feedforward neural network

        Args:
            args (parser.ArgumentParser): command line arguments
        """
        super().__init__()
        # input weights
        self.W_h = nn.Parameter(
            torch.randn(args.n_features, args.n_hidden) * args.weight_init
        )
        self.b_h = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * 1 / np.sqrt(args.n_hidden)
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h = None
        self.y_h = None
        self.y = None

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y


class ScaledNet(nn.Module):
    """Similar to Nnet above, but with separate params for input and context weights"""

    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        # input weights
        self.W_hin = torch.randn(args.n_features - 2, args.n_hidden) * args.weight_init
        self.W_hc = torch.randn(2, args.n_hidden) * args.ctx_w_init

        self.W_h = nn.Parameter(torch.cat((self.W_hin, self.W_hc), axis=0))
        self.b_h = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * args.weight_init
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h = None
        self.y_h = None
        self.y = None

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y


class ChoiceNet(nn.Module):
    """same as above, but with outputs passed through sigmoid"""

    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        # input weights
        self.W_h = nn.Parameter(
            torch.randn(args.n_features, args.n_hidden) * args.weight_init
        )
        self.b_h = nn.Parameter(torch.zeros(args.n_hidden))

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * 1 / np.sqrt(args.n_hidden)
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h = None
        self.y_h = None
        self.y = None

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = F.sigmoid(self.y_h @ self.W_o + self.b_o)

        return self.y


class Gatednet(nn.Module):
    """nnet with manual context gating"""

    def __init__(self, args: argparse.ArgumentParser):
        super().__init__()
        # input weights
        self.W_hin = torch.randn(args.n_features - 2, args.n_hidden) * args.weight_init
        self.W_hc = torch.ones(2, args.n_hidden, requires_grad=False)
        self.W_hc[0, 0::2] *= -1
        self.W_hc[1, 1::2] *= -1

        self.W_h = nn.Parameter(torch.cat((self.W_hin, self.W_hc), axis=0))
        self.b_h = nn.Parameter(torch.zeros(args.n_hidden))

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * args.weight_init
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h = None
        self.y_h = None
        self.y = None

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y
