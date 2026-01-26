# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
from torch import nn
from torch.nn.functional import leaky_relu


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return self.scale * leaky_relu(x + self.bias.reshape((1, -1, 1, 1))
            [:, :x.shape[1]], self.negative_slope, inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
