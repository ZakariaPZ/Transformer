from torch import nn 
from layers import LayerNorm, PositionWiseFFN, SinusoidalPositionalEncoding
from attention import Attention

class TransformerLayer(nn.Module):
    def __init__(
            self, 
            n_heads,
            d_model,
            d_hidden,
            dropout
        ):
        super().__init__()

        self.attention = Attention(n_heads, d_model, d_model, d_model, d_model)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.linear = PositionWiseFFN(d_model, d_hidden, dropout)
    
    def forward(self, x):
        # x is shape (seq_len, d_model)
        attention_output = self.attention(x)
        # attention_output is shape (seq_len, d_model)
        residual1 = self.layernorm1(attention_output) + x
        # residual1 is shape (seq_len, d_model)
        ffn_output = self.linear(residual1)
        # ffn_output is of shape (seq_len, d_model)
        output = self.layernorm2(ffn_output) + residual1
        return output