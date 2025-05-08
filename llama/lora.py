import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: float, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRALinear(nn.Linear, LoRALayer):
    """
    Implementation of LoRA-augmented Linear layer.
    LoRA decomposes a weight update into two low-rank matrices.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,  # rank of LoRA
        lora_alpha: float = 16,  # scaling factor
        lora_dropout: float = 0.0,  # dropout probability for LoRA layers
        merge_weights: bool = False,  # whether to merge weights during inference
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=False, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                         merge_weights=merge_weights)
        
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

def mark_only_lora_as_trainable(model: nn.Module):
    """
    Freeze all parameters in the model except for LoRA parameters.
    
    Args:
        model: The model containing LoRA layers
        
    Returns:
        Number of trainable parameters and total parameters
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA A and B matrices
    trainable_params = 0
    total_params = 0
    
    for module in model.modules():
        if isinstance(module, LoRALayer) and hasattr(module, 'lora_A'):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            
            trainable_params += module.lora_A.numel() + module.lora_B.numel()
        
        total_params += sum(p.numel() for p in module.parameters())
    
    return trainable_params, total_params

def apply_lora_to_llama(model, rank=8, alpha=16, dropout=0.05):
    """
    Apply LoRA to the Llama model by replacing Q and V projection layers.
    
    Args:
        model: Llama model
        rank: Rank of LoRA matrices
        alpha: Scaling factor for LoRA
        dropout: Dropout rate for LoRA layers
        
    Returns:
        Modified model with LoRA layers and percentage of trainable parameters
    """
    # Replace the Q and V projections in each attention block
    for layer in model.layers:
        # Replace Q projection
        in_features = layer.attention.wq.in_features
        out_features = layer.attention.wq.out_features
        q_linear = LoRALinear(
            in_features=in_features, 
            out_features=out_features,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout
        )
        # Copy the original weights to our LoRA linear layer
        q_linear.weight.data.copy_(layer.attention.wq.weight.data)
        layer.attention.wq = q_linear
        
        # Replace V projection
        in_features = layer.attention.wv.in_features
        out_features = layer.attention.wv.out_features
        v_linear = LoRALinear(
            in_features=in_features, 
            out_features=out_features,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout
        )
        # Copy the original weights to our LoRA linear layer
        v_linear.weight.data.copy_(layer.attention.wv.weight.data)
        layer.attention.wv = v_linear
    
    # Mark only LoRA parameters as trainable
    trainable_params, total_params = mark_only_lora_as_trainable(model)
    
    # Calculate percentage of trainable parameters
    percentage = 100 * trainable_params / total_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {percentage:.4f}%")
    
    return model, [percentage, trainable_params, total_params]