# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class SoftNLL(nn.Module):

    def __init__(self):
        """The `soft' version of negative_log_likelihood, where y is a distribution
                over classes rather than a one-hot coding
            """
        super(SoftNLL, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(torch.log(input) * target, dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
