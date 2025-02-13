from torch import nn 
from layers import LayerNorm, PositionWiseFFN
from attention import MultiHeadAttention
import torch


class DecoderLayer(nn.Module):
    def __init__(
            self, 
            n_heads,
            d_model,
            d_hidden,
            dropout
        ):
        super().__init__()

        self.attention = MultiHeadAttention(n_heads, d_model, d_model, d_model, d_model)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.linear = PositionWiseFFN(d_model, d_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x is shape (seq_len, d_model)
        attention_output = self.attention(x, mask)
        x = x + self.dropout(attention_output)
        # x is shape (seq_len, d_model)
        x = self.layernorm1(x)
        # x is shape (seq_len, d_model)
        ffn_output = self.linear(x)
        x = x + ffn_output
        # ffn_output is of shape (seq_len, d_model)
        x = self.layernorm2(x)
        return x

def create_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))
