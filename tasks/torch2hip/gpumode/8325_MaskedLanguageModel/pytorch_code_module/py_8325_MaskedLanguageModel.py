# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.optim
import torch.onnx.operators


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden': 4, 'vocab_size': 4}]
