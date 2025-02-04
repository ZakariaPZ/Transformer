import torch
from torch import nn
import math 


class Attention(nn.Module):
    
    def __init__(self, n_heads, d_model, d_q, d_k, d_v):
        super().__init__() 

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.softmax = nn.Softmax(dim=-1)  # Softmax along rows 

        # Projection matrices 
        # W is of shape (d_model, d_{q,k,v} * n_heads) 
        # Linear implements Y = XW^T + B
        self.W_q = nn.Linear(d_model, d_q * n_heads)  
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)
        self.W_o = nn.Linear(d_v * n_heads, d_model)

    def forward(self, x): 
        q = self.W_q(x).reshape(x.shape[0], x.shape[1], self.n_heads, self.d_q).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_q)
        k = self.W_k(x).reshape(x.shape[0], x.shape[1], self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        v = self.W_v(x).reshape(x.shape[0], x.shape[1], self.n_heads, self.d_v).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_v)

        scaling_factor = math.sqrt(self.d_q)  # Scaling factor for softmax

        # q @ k.T is of shape (batch_size, n_heads, seq_len, seq_len)
        # Multiplying with v is of shape (batch_size, n_heads, seq_len, d_v)
        y = self.softmax(q @ k.transpose(-1, -2) / scaling_factor) @ v
        y = y.transpose(1, 2).reshape(y.shape[0], y.shape[2], self.n_heads * self.d_v)  # (batch_size, seq_len, n_heads * d_v)
        return self.W_o(y)
