# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    The 'soft' version of negative log likelihood, where target is a distribution over classes.

    Args:
        input (torch.Tensor): Input tensor of probabilities, shape (..., num_classes)
        target (torch.Tensor): Target tensor of probabilities, same shape as input

    Returns:
        torch.Tensor: Scalar tensor, the mean negative log likelihood
    """
    # Compute elementwise log, multiply by target, sum over last dimension, mean over batch
    return -torch.mean(torch.sum(torch.log(input) * target, dim=1))

class SoftNLL(nn.Module):
    def __init__(self):
        """The `soft' version of negative_log_likelihood, where y is a distribution
        over classes rather than a one-hot coding
        """
        super(SoftNLL, self).__init__()

    def forward(self, input, target, fn=module_fn):
        return fn(input, target)

def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {}]
