# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return F.relu(x, inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
