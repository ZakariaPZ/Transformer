import torch
from torch import nn
import math 


class Attention(nn.Module):
    
    def __init__(self, d_model, d_q, d_k, d_v):
        super().__init__() 

        self.softmax = nn.Softmax(dim=1)  # Softmax along rows 

        # Projection matrices 
        self.W_q = nn.Linear(d_model, d_q)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scaling_factor = math.sqrt(q.shape[-1])

        return self.softmax(q @ k.T / scaling_factor) @ v
