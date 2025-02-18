import torch
from torch import nn
import math


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        # Define alpha and beta as learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(d_model)) 
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps 
    
    def forward(self, x):
        # Calculate mean and variance of activations of previous layer
        # Input is of shape (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        x = (x - mean) / (torch.sqrt(var) + 1e-6) # normalize activations
        x = self.gamma * x + self.beta  # scale and shift activations
        return x


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()

        # Input is of shape (batch_size, seq_len, d_model)
        # Output should have the same shape - we just want to improve the representational
        # power of the model by adding non-linear transformations
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Apply dropout after ReLU

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x 


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        positions = torch.arange(max_len).float().unsqueeze(1)  # (10000, 1)
        embedding_indices = torch.arange(0, d_model, 2).float().unsqueeze(0)  # (1, d_model)
        div_term = torch.exp(-(embedding_indices / d_model) * math.log(max_len))  # (1, d_model)

        pe = torch.zeros(max_len, d_model)
        # broadcast (10000, 1) and (1, d_model) to get (10000, d_model)
        pe[:, ::2] = torch.sin(positions * div_term)  
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return x
