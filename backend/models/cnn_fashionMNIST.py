"""A script that defines a simple CNN, used in the PyTorch example to solve
MNIST problem
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        print("Creating basic CNN MNIST model for MSN")
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        #self.act1 = nn.PReLU()
        self.act1 = nn.ReLU6()
        #self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        #self.act2 = nn.PReLU()
        self.act2 = nn.ReLU6()
        #self.act2 = nn.Tanh()
        self.fc1 = nn.Linear(1024, 64)
        #self.act3 = nn.PReLU()
        self.act3 = nn.ReLU6()
        #self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """Forward pass over the model."""
        x = F.max_pool2d(self.conv1(x), 2)
        x = self.act1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.act2(x)
        (_, C, H, W) = x.data.size()
        x = x.view(-1 , C*H*W)
        #print(C*H*W)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
