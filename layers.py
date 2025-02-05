import torch
from torch import nn
import math 

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        # define alpha and beta as learnable parameters
        self.gamma = torch.nn.Parameter(torch.tensor([1., 2., 3.])) 
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps 
    
    def forward(self, x):
        # calculate mean and variance of activations 
        # input is of shape (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        x = (x - mean) / (torch.sqrt(var) + 1e-6) # normalize activations
        x = self.gamma * x + self.beta  # scale and shift activations
        return x
