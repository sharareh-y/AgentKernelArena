# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class SoftmaxModule(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, v):
        return v.softmax(self.axis)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'axis': 4}]
