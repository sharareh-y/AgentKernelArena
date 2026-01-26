# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleMatmulModule(torch.nn.Module):

    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b):
        return a.matmul(b + b)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
