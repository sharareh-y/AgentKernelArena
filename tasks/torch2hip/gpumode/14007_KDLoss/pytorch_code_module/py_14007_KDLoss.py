# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.data


class KDLoss(nn.Module):

    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target) * self.temp_factor ** 2 / input.size(
            0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temp_factor': 4}]
