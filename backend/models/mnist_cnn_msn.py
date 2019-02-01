"""A script that defines a simple CNN, used in the PyTorch example to solve
MNIST problem
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        print("Creating basic CNN MNIST model for MSN")
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.act1 = nn.ReLU6()
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.act2 = nn.ReLU6()
        self.act2 = nn.Tanh()
        self.fc1 = nn.Linear(320, 50)
        #self.act3 = nn.ReLU6()
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = self.act1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.act2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
