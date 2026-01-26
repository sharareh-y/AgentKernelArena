# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    v: torch.Tensor,
    a: float,
    max: float,
) -> torch.Tensor:
    """
    Applies a scaled sigmoid activation and multiplies by max.

    Args:
        v (torch.Tensor): Input tensor.
        a (float): Scaling factor for input.
        max (float): Maximum value to scale the sigmoid output.

    Returns:
        torch.Tensor: Output tensor after scaled sigmoid and multiplication.
    """
    act = torch.sigmoid(a * v) * max
    return act

class Sigmoid(torch.nn.Module):
    def __init__(self, a=1, max=10):
        super().__init__()
        self.a = a
        self.max = max

    def forward(self, v, fn=module_fn):
        return fn(v, self.a, self.max)

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
