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
        #self.act1 = nn.PReLU()
        #self.act1 = nn.ReLU()
        #self.act1 = nn.Tanh()
        self.act1 = nn.Tanhshrink()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        #self.act2 = nn.PReLU()
        #self.act2 = nn.ReLU()
        #self.act2 = nn.ReLU6()
        #self.act2 = nn.Tanh()
        self.act2 = nn.Tanhshrink()
        self.fc1 = nn.Linear(1600, 100)
        #self.act3 = nn.PReLU()
        #self.act3 = nn.ReLU6()
        #self.act3 = nn.ReLU()
        #self.act3 = nn.Tanh()
        self.act3 = nn.Tanhshrink()
        self.fc2 = nn.Linear(100, 32)
        #self.act4 = nn.PReLU()
        #self.act4 = nn.ReLU6()
        #self.act4 = nn.ReLU()
        #self.act4 = nn.Tanh()
        self.act4 = nn.Tanhshrink()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        """Forward pass over the model."""
        #print(x[0][0:10])
        x = self.conv1(x)
        #print(x[0][0:10])
        x = self.clamp(x)
        #x = x.clamp(-0.1, 0.1)
        #print(x[0][0:10])
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.clamp(x)
        x = F.max_pool2d(x, 2)
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C*H*W)
        #print(C*H*W)
        x = self.fc1(x)
        x = self.clamp(x)
        x = self.fc2(x)
        x = self.clamp(x)
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        return x

    def clamp(self, x):
        #print(x[0][0:100])
        threshold = torch.full_like(x, 0.5)
        #threshold = 0.5
        selections = x.gt(threshold)
        x.fill_(0.)
        x[selections] = 0.5
        #print(x[0][0:100])
        return x
