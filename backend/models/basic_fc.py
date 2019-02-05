"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        print("\nCreating basic Fully-Connected model for Function Solving\n")
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        #self.act3 = nn.PReLU()
        #self.act3 = nn.ReLU6()
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, lower, upper):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        #x = F.log_softmax(x, dim=1)
        return torch.clamp(x, lower, upper)
