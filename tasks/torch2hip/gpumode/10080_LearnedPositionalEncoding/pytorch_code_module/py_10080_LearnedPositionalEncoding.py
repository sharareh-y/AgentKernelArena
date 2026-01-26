# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.optim


class LearnedPositionalEncoding(nn.Module):

    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 3200, 512))

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
        return x + position_embeddings


def get_inputs():
    return [torch.rand([4, 4, 3200, 512])]


def get_init_inputs():
    return [[], {'max_position_embeddings': 4, 'embedding_dim': 4,
        'seq_length': 4}]
