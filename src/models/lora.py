import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear layer with LoRA adapters.
    Original weight is frozen. Only A and B are trained.
    """
    def __init__(self, linear:nn.Linear, rank:int=8, alpha:float=16.0, dropout:float=0.0):
        super().__init__()
        self.linear = linear          # original frozen weight
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        #disable gradient for the original weight
        self.linear.weight.requires_grad_(mode=False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(mode=False)

        in_features = linear.in_features
        out_features = linear.out_features

        #low-rank matrices A and B
        self.A = nn.Parameter(torch.zeros(in_features, rank))  # (in_features, rank)
        self.B = nn.Parameter(torch.zeros(rank, out_features)) # (rank, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        #initialize A and B weights
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) #same initialization as nn.Linear for A
        nn.init.zeros_(self.B) #B is initialized to zero so that the LoRA layer starts as the original linear layer (no change), and then learns the low-rank update during training.

    def forward(self, x):
        #original frozen output
        frozen_out = self.linear(x) # (batch, seq_len, out_features)

        #LoRA update
        lora_update = self.dropout(torch.matmul(torch.matmul(x, self.A), self.B)) * self.scaling # (x @ A) @ B, then scaled by alpha/rank

        return frozen_out + lora_update
    
    def merge(self):
        """
        Merge LoRA weights into the original linear layer (for inference speed).
        After merging, the layer becomes a standard nn.Linear.
        """
        merged_weight = self.linear.weight + (self.A @ self.B).T * self.scaling # (out_features, in_features) + (out_features, rank) @ (rank, in_features)
        self.linear.weight = nn.Parameter(merged_weight, requires_grad=False) #freeze the merged weight
        #clear the lora parameters to save memory
        self.A = None
        self.B = None

def inject_lora(model, target_modules=None, rank=8, alpha=16.0, dropout=0.0):
    """
    Replace nn.Linear layers in the model with LoRALinear wrappers.
    target_modules: list of module names to apply LoRA (e.g., ['q', 'v']).
    If None, apply to all nn.Linear layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and (target_modules is None or any(t in name for t in target_modules)):
            setattr(model, name, LoRALinear(module, rank, alpha, dropout)) #replace the linear layer with a LoRALinear wrapper
        elif isinstance(module, nn.Module):
            inject_lora(module, target_modules, rank, alpha, dropout) #recursively apply to child modules

def save_lora_state(model, path):
    """
    Save only the LoRA parameters (A, B matrices) from the model.
    """
    lora_params = {name:param for name, param in model.named_parameters() if 'A' in name or 'B' in name}
    torch.save(lora_params, path)

def load_lora_state(model, path):
    """
    Load LoRA parameters from a file and update the model's LoRA layers.
    """
    lora_params = torch.load(path)
    model_state = model.state_dict()
    for name, param in lora_params.items():
        if name in model_state:
            model_state[name].copy_(param) #update the LoRA parameters in the model