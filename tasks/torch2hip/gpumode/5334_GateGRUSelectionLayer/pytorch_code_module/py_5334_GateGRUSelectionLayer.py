# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class GateGRUSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateGRUSelectionLayer, self).__init__()
        self.reset = nn.Linear(dim_model * 2, dim_model)
        self.update = nn.Linear(dim_model * 2, dim_model)
        self.proposal = nn.Linear(dim_model * 2, dim_model)

    def forward(self, x_1, x_2, *args):
        reset = torch.sigmoid(self.reset(torch.cat([x_1, x_2], -1)))
        update = torch.sigmoid(self.update(torch.cat([x_1, x_2], -1)))
        proposal = torch.tanh(self.proposal(torch.cat([reset * x_1, x_2], -1)))
        out = (1 - update) * x_1 + update * proposal
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'dim_ff': 4, 'prob_dropout': 0.5}]
