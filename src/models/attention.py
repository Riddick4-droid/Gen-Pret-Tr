import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    the usual scaled dot-product attention mechanism.
    Args:
        query: (batch, num_heads, seq_len, d_k)
        key:   (batch, num_heads, seq_len, d_k)
        value: (batch, num_heads, seq_len, d_v)
        mask:  (batch, 1, seq_len, seq_len) or broadcastable, True = masked
        dropout: nn.Dropout or None
    Returns:
        output: (batch, num_heads, seq_len, d_v)
        attn_weights: (batch, num_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi‑head self‑attention with optional causal masking.
    note that the d_model must be divisible by num_heads, and the attention is computed 
    in parallel for all heads, then concatenated and projected back to d_model.
     The causal_mask is used to prevent attending to future tokens during training, ensuring the autoregressive property of the model.
     The attention weights are returned for visualization or analysis purposes.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model #d
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        """
        x: (batch, seq_len, d_model)
        causal_mask: (batch, 1, seq_len, seq_len) or None
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        B, T, _ = x.shape #(batch, seq_len, d_model)

        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=causal_mask, dropout=self.dropout
        )

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Final linear projection
        output = self.W_o(attn_output)
        return output, attn_weights