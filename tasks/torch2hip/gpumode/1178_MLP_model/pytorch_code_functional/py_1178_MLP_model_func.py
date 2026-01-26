# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Functional kernel for the 7-layer MLP
def mlp_kernel(
    xb: torch.Tensor,
    linear1_weight: torch.Tensor, linear1_bias: torch.Tensor,
    linear2_weight: torch.Tensor, linear2_bias: torch.Tensor,
    linear3_weight: torch.Tensor, linear3_bias: torch.Tensor,
    linear4_weight: torch.Tensor, linear4_bias: torch.Tensor,
    linear5_weight: torch.Tensor, linear5_bias: torch.Tensor,
    linear6_weight: torch.Tensor, linear6_bias: torch.Tensor,
    linear7_weight: torch.Tensor, linear7_bias: torch.Tensor,
) -> torch.Tensor:
    # Flatten input
    xb = xb.view(xb.size(0), -1)
    # Layer 1
    out = F.linear(xb, linear1_weight, linear1_bias)
    out = F.relu(out)
    # Layer 2
    out = F.linear(out, linear2_weight, linear2_bias)
    out = F.relu(out)
    # Layer 3
    out = F.linear(out, linear3_weight, linear3_bias)
    out = F.relu(out)
    # Layer 4
    out = F.linear(out, linear4_weight, linear4_bias)
    out = F.relu(out)
    # Layer 5
    out = F.linear(out, linear5_weight, linear5_bias)
    out = F.relu(out)
    # Layer 6
    out = F.linear(out, linear6_weight, linear6_bias)
    out = F.relu(out)
    # Output layer
    out = F.linear(out, linear7_weight, linear7_bias)
    return out

# Functional module implementation
def module_fn(
    xb: torch.Tensor,
    linear1_weight: torch.Tensor, linear1_bias: torch.Tensor,
    linear2_weight: torch.Tensor, linear2_bias: torch.Tensor,
    linear3_weight: torch.Tensor, linear3_bias: torch.Tensor,
    linear4_weight: torch.Tensor, linear4_bias: torch.Tensor,
    linear5_weight: torch.Tensor, linear5_bias: torch.Tensor,
    linear6_weight: torch.Tensor, linear6_bias: torch.Tensor,
    linear7_weight: torch.Tensor, linear7_bias: torch.Tensor,
) -> torch.Tensor:
    return mlp_kernel(
        xb,
        linear1_weight, linear1_bias,
        linear2_weight, linear2_bias,
        linear3_weight, linear3_bias,
        linear4_weight, linear4_bias,
        linear5_weight, linear5_bias,
        linear6_weight, linear6_bias,
        linear7_weight, linear7_bias,
    )

class MLP_model(nn.Module):
    """Feedfoward neural network with 6 hidden layer"""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        self.linear7 = nn.Linear(32, out_size)

    def forward(self, xb, fn=module_fn):
        return fn(
            xb,
            self.linear1.weight, self.linear1.bias,
            self.linear2.weight, self.linear2.bias,
            self.linear3.weight, self.linear3.bias,
            self.linear4.weight, self.linear4.bias,
            self.linear5.weight, self.linear5.bias,
            self.linear6.weight, self.linear6.bias,
            self.linear7.weight, self.linear7.bias,
        )

def get_inputs():
    return [torch.rand([4, 4])]

def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
