import os
import yaml
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def load_config(config_path:str) -> dict:
    """Loads YAML configuration file and returns a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_model_info(model: torch.nn.Module) -> dict:
    """Returns a dictionary with model parameter count and size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_params * 4 / (1024 ** 2)  # assuming float32
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_size_mb": total_size_mb
    }
def get_resource_info()->dict:
    """"returns info about the compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "total_memory_gb": total_memory
        }
    else:
        return {
            "device": "cpu"
        }
    
def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    """Saves model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

#attention mask visualization
def create_combined_mask(seq_len:int, pad_mask:torch.Tensor=None) -> torch.Tensor:
    """
    Returns a boolean mask of shape (seq_len, seq_len) where True = ignore.
    Combines causal (upper triangular) with optional padding mask.
    """
    #get or set causal mask
    causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # (seq_len, seq_len)
    if pad_mask is not None:
        #pad mask: (seq_len,)-> with 0 = padding
        pad = (pad_mask == 0).unsqueeze(0)
        causal = causal | pad  # combine masks
    return causal

def plot_mask(mask: torch.Tensor, title: str = "Attention Mask", annotate: bool = True):
    """
    Plot a boolean attention mask with optional cell annotations (T/F).
    """
    seq_len = mask.shape[0]
    plt.figure(figsize=(max(6, seq_len*0.6), max(5, seq_len*0.5)))
    plt.imshow(mask, cmap='gray_r', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    if annotate:
        for i in range(seq_len):
            for j in range(seq_len):
                val = "T" if mask[i, j] else "F"
                color = "white" if mask[i, j] else "black"
                plt.text(j, i, val, ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(label="True = masked")
    plt.show()

def plot_attention_scores_after_masking(seq_len: int, pad_mask: torch.Tensor = None):
    """
    Generates random attention scores, applies the combined mask,
    and plots the resulting matrix with -inf shown.
    """
    scores = torch.randn(seq_len, seq_len) * 2
    mask = create_combined_mask(seq_len, pad_mask)
    scores_masked = scores.masked_fill(mask, float('-inf'))
    arr = scores_masked.numpy()
    arr_vis = np.where(np.isneginf(arr), -1e9, arr)

    plt.figure(figsize=(max(6, seq_len*0.6), max(5, seq_len*0.5)))
    plt.imshow(arr_vis, cmap='viridis', interpolation='nearest')
    plt.title("Attention scores after masking (-inf shown)")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    for i in range(seq_len):
        for j in range(seq_len):
            if np.isneginf(arr[i, j]):
                text = "-inf"
                color = "white"
            else:
                text = f"{arr[i, j]:.1f}"
                color = "black"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=7)
    plt.colorbar()
    plt.show()