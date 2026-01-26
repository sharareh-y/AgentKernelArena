# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the ReLU activation function in-place.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying in-place ReLU.
    """
    return F.relu(x, inplace=True)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
