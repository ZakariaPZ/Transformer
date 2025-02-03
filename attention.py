import torch
from torch import nn

class Attention(nn.Module):
    
    def __init__(self):
        super().__init__() 

        self.softmax = nn.Softmax(dim=1)  # Softmax along rows 

    def forward(self, x):
        return self.softmax(x @ x.T)
    
# Tests
# 1. Check that rows sum to 1 (softmax)
# 2. Check that the output is same shape as input
# 3. Check that the output is symmetric

x = torch.randn(3, 3)
print(x)
attention = Attention()
print(attention(x))