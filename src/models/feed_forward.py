import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    """
    Two‑layer fully connected network with GELU activation.
    can be more than 2 layers, but 2 is standard in the original transformer paper.
    more than two increases the model capacity but also the computational cost, so we stick to 2 for simplicity.
    Applied independently to each position in the sequence.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) #d_ff is the hidden dimension of the feed‑forward network, typically larger than d_model (e.g., 4*d_model) to increase capacity.
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x #logits after the feed‑forward network, no softmax here since this is not the final output layer, but an intermediate transformation within the decoder block.