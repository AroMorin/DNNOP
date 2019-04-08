"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.act1 = nn.PReLU()
        #self.act1 = nn.ReLU6()
        #self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, observation):
        """Forward pass over the model."""
        origin = observation[0]
        x1_low = observation[1][0]
        x2_low = observation[1][1]
        x1_high = observation[2][0]
        x2_high = observation[2][1]
        x = self.fc1(origin)
        x = self.act1(x)
        x = self.fc2(x)
        x[0] = torch.clamp(x[0], x1_low, x1_high)
        x[1] = torch.clamp(x[1], x2_low, x2_high)
        return x
