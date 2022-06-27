import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse


class Nnet(nn.Module):
    """simple feed forward neural network with a single hidden layer"""

    def __init__(self, args: argparse.Namespace):
        """constructor for feedforward neural network

        Args:
            args (parser.Namespace): command line arguments
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
        """performs forward pass through network

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y


class ScaledNet(nn.Module):
    """Similar to Nnet above, but with separate params for input and context weights"""

    def __init__(self, args: argparse.Namespace):
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
        """performs forward pass through network

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y


class ScaledNet2Hidden(nn.Module):
    """simple feedforward mlp with two hidden layers"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # 1st hidden layer:
        self.W_h1_in = (
            torch.randn(args.n_features - 2, args.n_hidden) * args.weight_init
        )
        self.W_h1_c = torch.randn(2, args.n_hidden) * args.ctx_w_init
        self.W_h1 = nn.Parameter(torch.cat((self.W_h1_in, self.W_h1_c), axis=0))
        self.b_h1 = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # 2nd hidden layer:
        self.W_h2 = nn.Parameter(
            torch.randn(args.n_hidden, args.n_hidden) * args.weight_init
        )
        self.b_h2 = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * args.weight_init
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h1 = None
        self.y_h1 = None
        self.x_h2 = None
        self.y_h2 = None
        self.y = None

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """performs forward pass through network

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """

        self.x_h1 = x_batch @ self.W_h1 + self.b_h1
        self.y_h1 = F.relu(self.x_h1)

        self.x_h2 = self.y_h1 @ self.W_h2 + self.b_h2
        self.y_h2 = F.relu(self.x_h2)
        self.y = self.y_h2 @ self.W_o + self.b_o
        return self.y


class ScaledNet2Hidden2Ctx(nn.Module):
    """same as ScaledNet2Hidden, but with context signal acting directly on each hidden layer,
    similar to Musslick et al - style architectures"""

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # 1st hidden layer:
        self.W_h1_in = (
            torch.randn(args.n_features - 2, args.n_hidden) * args.weight_init
        )
        self.W_h1_c = torch.randn(2, args.n_hidden) * args.ctx_w_init
        self.W_h1 = nn.Parameter(torch.cat((self.W_h1_in, self.W_h1_c), axis=0))
        self.b_h1 = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # 2nd hidden layer:
        self.W_h2_in = torch.randn(args.n_hidden, args.n_hidden) * args.weight_init
        self.W_h2_c = torch.randn(2, args.n_hidden) * args.ctx_w_init
        self.W_h2 = nn.Parameter(torch.cat((self.W_h2_in, self.W_h2_c), axis=0))
        self.b_h2 = nn.Parameter(torch.zeros(args.n_hidden)) + 0.1

        # output weights
        self.W_o = nn.Parameter(
            torch.randn(args.n_hidden, args.n_out) * args.weight_init
        )
        self.b_o = nn.Parameter(torch.zeros(args.n_out))

        self.x_h1 = None
        self.y_h1 = None
        self.x_h2 = None
        self.y_h2 = None
        self.y = None
        self.n_features = args.n_features

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """performs forward pass through network, context signal applied separately
        to each hidden layer

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """
        x_batch = torch.reshape(x_batch, (-1, self.n_features))        
        self.x_h1 = x_batch @ self.W_h1 + self.b_h1
        self.y_h1 = F.relu(self.x_h1)
        
        self.x_h2 = (
            torch.cat((self.y_h1, x_batch[:, -2:]), axis=1) @ self.W_h2 + self.b_h2
        )
        self.y_h2 = F.relu(self.x_h2)
        self.y = self.y_h2 @ self.W_o + self.b_o
        return self.y


class ChoiceNet(nn.Module):
    """same as above, but with outputs passed through sigmoid"""

    def __init__(self, args: argparse.Namespace):
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
        """performs forward pass through network

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = F.sigmoid(self.y_h @ self.W_o + self.b_o)

        return self.y


class Gatednet(nn.Module):
    """nnet with manual context gating"""

    def __init__(self, args: argparse.Namespace):
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
        """performs forward pass through network

        Args:
            x_batch (torch.Tensor): input data, single datum or (mini)batch

        Returns:
            torch.Tensor: the network's prediction
        """
        self.x_h = x_batch @ self.W_h + self.b_h
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o

        return self.y
