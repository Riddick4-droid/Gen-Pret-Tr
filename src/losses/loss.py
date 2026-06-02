import torch
import torch.nn as nn

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100,
                 label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Cross‑entropy loss for next‑token prediction.
    If label_smoothing > 0, applies label smoothing regularisation.
    presumes logits are raw (no softmax) and of shape (B, T, vocab_size), labels are (B, T).
    ignore_index tokens are not included in loss or perplexity calculations.
    also supports label smoothing which can help regularize the model by preventing it from becoming too confident in its predictions.
    """
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

def compute_perplexity(loss: float) -> float:
    """Perplexity is exp(cross‑entropy loss). Returns inf if loss is too large to prevent overflow."""
    import math
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')