import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Custom Layer Normalisation .
    Normalises the last dimension of the input (d_model).
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps # small constant to prevent division by zero
        self.gamma = nn.Parameter(torch.ones(d_model)) # learnable scale parameter initialized to 1
        self.beta  = nn.Parameter(torch.zeros(d_model)) # learnable shift parameter initialized to 0

    def forward(self, x):
        """normalize the last dimension of x and apply learnable scale and shift."""
        # x: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta