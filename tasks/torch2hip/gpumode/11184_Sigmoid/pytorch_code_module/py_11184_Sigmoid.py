# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class Sigmoid(torch.nn.Module):

    def __init__(self, a=1, max=10):
        super().__init__()
        self.a = a
        self.max = max

    def forward(self, v):
        sig = nn.Sigmoid()
        act = sig(self.a * v) * self.max
        return act


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
