# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    position_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Adds learned positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, heads, seq_len, embed_dim)
        position_embeddings (torch.Tensor): Positional embedding tensor of shape (1, seq_len, embed_dim)

    Returns:
        torch.Tensor: Output tensor after adding positional embeddings
    """
    return x + position_embeddings

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 3200, 512))

    def forward(self, x, position_ids=None, fn=module_fn):
        return fn(x, self.position_embeddings)

def get_inputs():
    return [torch.rand([4, 4, 3200, 512])]

def get_init_inputs():
    return [[], {'max_position_embeddings': 4, 'embedding_dim': 4, 'seq_length': 4}]
