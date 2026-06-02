import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.feed_forward import PositionWiseFeedForward
from src.models.layer_norm import LayerNorm

class TransformerDecoderBlock(nn.Module):
    """
    A single decoder layer with pre‑layer norm, masked self‑attention,
    feed‑forward network, and residual connections.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        """
        x: (batch, seq_len, d_model)
        causal_mask: (batch, 1, seq_len, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        # self‑attention with residual
        attn_out, attn_weights = self.self_attn(self.norm1(x), causal_mask)
        x = x + self.dropout(attn_out) #residual connection after attention

        # feed‑forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, attn_weights #x is logits after the decoder block, attn_weights can be used for visualization or analysis of attention patterns