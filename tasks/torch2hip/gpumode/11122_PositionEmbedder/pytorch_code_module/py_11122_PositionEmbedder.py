# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn


class PositionEmbedder(torch.nn.Module):
    """
    [batch_size, seq_length, embedding_size]
    """

    def __init__(self, max_sequence_length: 'int', embedding_dim: 'int'):
        super(PositionEmbedder, self).__init__()
        self.embedding = torch.nn.Embedding(max_sequence_length,
            embedding_dim, padding_idx=None)

    def forward(self, input_embeddings: 'torch.Tensor'):
        positions = torch.arange(input_embeddings.shape[1], device=
            input_embeddings.device)
        return input_embeddings + self.embedding(positions)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_sequence_length': 4, 'embedding_dim': 4}]
