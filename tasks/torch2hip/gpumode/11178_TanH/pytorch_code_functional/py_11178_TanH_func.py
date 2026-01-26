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
    Applies scaled tanh activation: tanh(a * v) * max

    Args:
        v (torch.Tensor): Input tensor
        a (float): Scaling factor for input
        max (float): Scaling factor for output

    Returns:
        torch.Tensor: Activated tensor
    """
    act = torch.tanh(a * v) * max
    return act


class TanH(torch.nn.Module):
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
