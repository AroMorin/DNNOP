"""A script that defines a simple CNN, used in the PyTorch example to solve
MNIST problem
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        print("Creating Object Detection model for MSN")
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        #self.act1 = nn.PReLU()
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        #self.act2 = nn.PReLU()
        #self.act2 = nn.ReLU6()
        self.act2 = nn.Tanh()
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        #self.act3 = nn.PReLU()
        #self.act3 = nn.ReLU6()
        self.act3 = nn.Tanh()
        self.conv4 = nn.Conv2d(30, 20, kernel_size=3)
        self.conv5 = nn.Conv2d(20, 10, kernel_size=3)
        self.fc1 = nn.Linear(1040, 400)
        #self.act4 = nn.PReLU()
        #self.act4 = nn.ReLU6()
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(400, 50)
        #self.act5 = nn.PReLU()
        #self.act5 = nn.ReLU6()
        self.act5 = nn.Tanh()
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        """Forward pass over the model."""
        x = F.max_pool2d(self.conv1(x), 2)
        x = self.act1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.act2(x)
        x = F.max_pool2d(self.conv3(x), 2)
        x = self.act3(x)
        x = F.max_pool2d(self.conv4(x), 2)
        x = self.act3(x)
        x = F.max_pool2d(self.conv5(x), 2)
        x = self.act3(x)
        (_, C, H, W) = x.data.size()
        x = x.view(-1 , C*H*W)
        #x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.fc3(x)
        # Normalized coordinates
        x1 = torch.clamp(x[0], 0, 1)
        x2 = torch.clamp(x[1], 0, 1)
        return [x1, x2]
