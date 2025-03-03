from torch import nn 
from layers import LayerNorm, PositionWiseFFN, SinusoidalPositionalEncoding, MultiHeadAttention
import torch
import math


class DecoderLayer(nn.Module):
    def __init__(
            self, 
            n_heads,
            d_head,
            d_model,
            d_hidden,
            dropout
        ):
        super().__init__()

        self.attention = MultiHeadAttention(n_heads, d_head, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.linear = PositionWiseFFN(d_model, d_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x is shape (N, d_model)
        residual = x
        x = self.attention(self.layer_norm1(x), mask)
        x = residual + self.dropout(x)

        # x is shape (N, d_model)
        residual = x
        x = self.linear(self.layer_norm2(x)) 
        x = residual + self.dropout(x)
        
        # ffn_output is of shape (N, d_model)
        return x
    

class Decoder(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            d_model, 
            d_head,
            d_hidden,
            n_heads,
            n_layers,
            dropout=0.1,
            max_len=10000
        ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        # Note: Can't use a normal list because it won't register the decoder layers
        # as modules of the Decoder class 
        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, d_head, d_model, d_hidden, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = LayerNorm(d_model)
        self.out_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x is of shape (batch_size, seq_len), where each token is represented by an integer
        # to be converted to an embedding by nn.Embedding. Furthermore, we multiply by 
        # sqrt(d_model) because the embedding layer's weights are typically initialized with 
        # a variance of 1/d_model, while the positional encondings have values in the range 
        # [-1, 1]. Without scaling, the positional embeddings would have a much larger 
        # influence on the sum than the embeddings.
        x = self.embedding(x) * math.sqrt(self.d_model) 

        # Add position information 
        x = self.positional_encoding(x) 
        x = self.dropout(x)

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return self.out_projection(x)  # Return logits because Torch's BCE loss includes softmax 
    
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
