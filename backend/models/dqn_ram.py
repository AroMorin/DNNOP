"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):

    def __init__(self, in_features, outputs):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(in_features, 256)
        self.act1 = nn.Relu()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.Relu()
        self.fc3 = nn.Linear(128, 64)
        self.act3 = nn.Relu()
        self.fc4 = nn.Linear(64, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x
