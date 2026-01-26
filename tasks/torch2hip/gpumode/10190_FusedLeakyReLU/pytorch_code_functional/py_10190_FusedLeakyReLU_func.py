# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_leaky_relu_fn(
    x: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float,
    scale: float,
) -> torch.Tensor:
    # Add bias (reshaped), slice to match input channels, apply leaky_relu, then scale
    x = x + bias.reshape(1, -1, 1, 1)[:, :x.shape[1]]
    x = F.leaky_relu(x, negative_slope=negative_slope, inplace=True)
    x = x * scale
    return x

def module_fn(
    x: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float,
    scale: float,
) -> torch.Tensor:
    return fused_leaky_relu_fn(x, bias, negative_slope, scale)

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x, fn=module_fn):
        return fn(x, self.bias, self.negative_slope, self.scale)

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {'channel': 4}]
