# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication of a with (b + b).

    Args:
        a (torch.Tensor): Input tensor of shape (..., M, K)
        b (torch.Tensor): Input tensor of shape (..., K, N)

    Returns:
        torch.Tensor: Output tensor of shape (..., M, N)
    """
    return torch.matmul(a, b + b)

class SimpleMatmulModule(nn.Module):
    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b, fn=module_fn):
        return fn(a, b)

def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
