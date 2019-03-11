"""A script that defines a simple FC model for nao robot applications"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.act1 = nn.PReLU()
        #self.act1 = nn.ReLU6()
        #self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """Forward pass over the model."""
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
