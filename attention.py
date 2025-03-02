from torch import nn
import math 


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, d_head, d_model):
        super().__init__() 

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head

        # Each head will have dimension d_heads, and we create n_heads of them
        # The total dimension will be d_heads * n_heads for query, key and value projections
        self.W_q = nn.Linear(d_model, d_head * n_heads)  
        self.W_k = nn.Linear(d_model, d_head * n_heads)
        self.W_v = nn.Linear(d_model, d_head * n_heads)
        self.W_o = nn.Linear(d_head * n_heads, d_model)

        self.softmax = nn.Softmax(dim=-1)  # Softmax along rows 

    def forward(self, x, mask=None): 
        B, N, _ = x.size()
        q = self.W_q(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, N, d_head)
        k = self.W_k(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, N, d_head)
        v = self.W_v(x).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, n_heads, N, d_head)

        scaling_factor = math.sqrt(self.d_head)  # Scaling factor for softmax
        attention_logits = q @ k.transpose(-1, -2) / scaling_factor

        if mask is not None:
            # Mask should be lower triangular - anything above the diagonal is 0
            # to prevent tokens from attending to future tokens.
            attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))

        attention_weights = self.softmax(attention_logits)
        y = attention_weights @ v  # (N, N) @ (B, n_heads, N, d_head) -> (B, n_heads, N, d_head)
        y = y.transpose(1, 2).reshape(B, N, self.n_heads * self.d_head) 

        return self.W_o(y)  # (B, N, d_model)
