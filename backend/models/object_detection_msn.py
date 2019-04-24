"""A script that defines a simple CNN, used in the PyTorch example to solve
MNIST problem
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        print("Creating Object Detection model for MSN")
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.act1 = nn.PReLU()
        #self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.act2 = nn.PReLU()
        #self.act2 = nn.ReLU6()
        #self.act2 = nn.Tanh()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.act3 = nn.PReLU()
        #self.act3 = nn.ReLU6()
        #self.act3 = nn.Tanh()
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3)
        self.act4 = nn.PReLU()
        #self.act4 = nn.Tanh()
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3)
        self.act5 = nn.PReLU()
        #self.act5 = nn.Tanh()
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3)
        self.act6 = nn.PReLU()
        #self.act6 = nn.Tanh()
        self.fc1 = nn.Linear(120, 64)
        self.act7 = nn.PReLU()
        #self.act7 = nn.ReLU6()
        self.act7 = nn.Tanh()
        self.fc2 = nn.Linear(64, 16)
        self.act8 = nn.PReLU()
        #self.act8 = nn.ReLU6()
        #self.act8 = nn.Tanh()
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        """Forward pass over the model."""
        x = F.max_pool2d(self.conv1(x), 2)
        x = self.act1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = self.act2(x)
        x = F.max_pool2d(self.conv3(x), 2)
        x = self.act3(x)
        x = F.max_pool2d(self.conv4(x), 2)
        x = self.act4(x)
        x = F.max_pool2d(self.conv5(x), 2)
        x = self.act5(x)
        x = F.max_pool2d(self.conv6(x), 2)
        x = self.act6(x)
        (_, C, H, W) = x.data.size()
        x = x.view(-1 , C*H*W)
        #print(C*H*W)
        x = self.fc1(x)
        x = self.act7(x)
        x = self.fc2(x)
        x = self.act8(x)
        x = self.fc3(x)
        # Normalized coordinates
        x = torch.clamp(x[:], 0, 1)
        return x
