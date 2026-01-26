# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
from torch import nn
import torch.onnx


class Gather(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]
        super().__init__()

    def forward(self, input: 'torch.Tensor', indices: 'torch.Tensor'):
        selection = self.selection + [indices]
        return input.__getitem__(selection)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
