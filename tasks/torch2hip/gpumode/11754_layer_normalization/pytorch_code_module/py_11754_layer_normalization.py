# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-08):
        """Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        """
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
