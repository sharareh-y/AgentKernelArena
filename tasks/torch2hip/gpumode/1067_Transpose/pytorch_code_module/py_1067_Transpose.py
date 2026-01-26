# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class Transpose(nn.Module):

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return input.transpose(self.dim1, self.dim2).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'between=' + str(self.dim1
            ) + ',' + str(self.dim2) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
