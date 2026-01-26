# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def silu_fn(x: torch.Tensor) -> torch.Tensor:
    # Applies the SiLU activation function: x * sigmoid(x)
    return x * torch.sigmoid(x)

def module_fn(x: torch.Tensor) -> torch.Tensor:
    # Equivalent functional implementation of SiLU
    return silu_fn(x)

class SiLU(nn.Module):
    @staticmethod
    def forward(x, fn=module_fn):
        return fn(x)

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
