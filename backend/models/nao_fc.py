"""A script that defines a simple FC model for nao robot applications"""
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2048)
        #self.act1 = nn.ReLU6()
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(2048, 24)

    def forward(self, x):
        """Forward pass over the model."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
