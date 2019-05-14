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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.act1 = nn.PReLU()
        #self.act1 = nn.ReLU6()
        #self.act1 = nn.ReLU()
        #self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.act2 = nn.PReLU()
        #self.act2 = nn.ReLU()
        #self.act2 = nn.ReLU6()
        #self.act2 = nn.Tanh()
        self.fc1 = nn.Linear(1600, 400)
        self.act3 = nn.PReLU()
        #self.act3 = nn.ReLU6()
        #self.act3 = nn.ReLU()
        #self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(400, 84)
        self.act4 = nn.PReLU()
        #self.act4 = nn.ReLU6()
        #self.act4 = nn.ReLU()
        #self.act4 = nn.Tanh()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass over the model."""
        x = self.conv1(x)
        x = self.act1(x)
        x = F.max_pool2d(x, 2)
        #x = torch.sin(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)
        #x = torch.sin(x)
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C*H*W)
        #print(C*H*W)
        x = self.fc1(x)
        x = self.act3(x)
        #x = torch.sin(x)
        x = self.fc2(x)
        x = self.act4(x)
        #x = torch.sin(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
