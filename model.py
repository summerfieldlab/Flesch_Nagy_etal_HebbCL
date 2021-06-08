import torch
import torch.utils.data
from torch import nn 
from torch.nn import functional as F 



class Nnet(nn.Module):
    def __init__(self,args):
        super().__init__()
        # input weights 
        self.W_h = nn.Parameter(torch.randn(args.n_features,args.n_hidden)*args.weight_init)
        self.b_h = nn.Parameter(torch.zeros(args.n_hidden))

        # output weights 
        self.W_o = nn.Parameter(torch.randn(args.n_hidden,args.n_out)*args.weight_init)
        self.b_o = nn.Parameter(torch.zeros(args.n_out))
        
            
    def forward(self,x_batch):        
        self.x_h = x_batch @ self.W_h + self.b_h 
        self.y_h = F.relu(self.x_h)
        self.y = self.y_h @ self.W_o + self.b_o
               
        return self.y

     

