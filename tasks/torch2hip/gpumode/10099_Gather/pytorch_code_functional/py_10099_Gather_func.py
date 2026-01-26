# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_fn(input: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Gathers values along an axis specified by dim using indices.

    Args:
        input (torch.Tensor): The source tensor.
        indices (torch.Tensor): The indices of elements to gather.
        dim (int): The axis along which to index.

    Returns:
        torch.Tensor: The result of gathering.
    """
    # Expand indices to match input shape for advanced indexing
    selection = [slice(None) for _ in range(dim)] + [indices]
    return input.__getitem__(selection)

def module_fn(input: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Functional implementation of the Gather module.

    Args:
        input (torch.Tensor): The source tensor.
        indices (torch.Tensor): The indices of elements to gather.
        dim (int): The axis along which to index.

    Returns:
        torch.Tensor: The result of gathering.
    """
    return gather_fn(input, indices, dim)

class Gather(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]

    def forward(self, input: torch.Tensor, indices: torch.Tensor, fn=module_fn):
        return fn(input, indices, self.dim)

def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]

def get_init_inputs():
    return [[], {}]
