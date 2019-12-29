import torch
from torch import nn


class DNN(nn.Module):
    def __init__(self, hidden_size=256):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))
