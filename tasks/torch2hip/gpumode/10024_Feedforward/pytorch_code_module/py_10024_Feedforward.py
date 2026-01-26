# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size=100):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        inp = torch.vstack([x, y])
        hidden = self.fc1(inp)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
