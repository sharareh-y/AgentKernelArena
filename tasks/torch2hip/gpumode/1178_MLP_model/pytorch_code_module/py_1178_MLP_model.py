# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_model(nn.Module):
    """Feedfoward neural network with 6 hidden layer"""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        self.linear7 = nn.Linear(32, out_size)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = F.relu(out)
        out = self.linear7(out)
        return out

    def training_step(self, batch, criterion):
        images, labels = batch
        out = self(images)
        loss = criterion(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        None

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
