# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    v: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """
    Applies softmax along the specified axis.

    Args:
        v (torch.Tensor): Input tensor.
        axis (int): Axis along which softmax is computed.

    Returns:
        torch.Tensor: Output tensor after softmax.
    """
    return F.softmax(v, dim=axis)

class SoftmaxModule(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, v, fn=module_fn):
        return fn(v, self.axis)

def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]

def get_init_inputs():
    return [[], {'axis': 4}]
