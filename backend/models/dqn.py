"""A script that defines a simple FC model for function solving"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        nb_channels = model_params['in channels']
        actions = model_params['number of outputs']
        self.conv1 = nn.Conv2d(nb_channels, 32, kernel_size=8,
                                stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.fc1 = nn.Linear(128, 32)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(32, actions)

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in channels": 3,
                            "number of outputs": 18,
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params

    def forward(self, x):
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        x = self.conv1(x)
        x = self.clamp(x)
        x = F.max_pool2d(x, 2)
        #x = self.act1(x)
        x = self.conv2(x)
        x = self.clamp(x)
        x = F.max_pool2d(x, 2)
        #x = self.act2(x)
        x = self.conv3(x)
        x = self.clamp(x)
        x = F.max_pool2d(x, 2)
        #x = self.act3(x)
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C*H*W)
        #print(C*H*W)
        x = self.fc1(x)
        x = self.clamp(x)
        #x = self.act4(x)
        x = self.fc2(x)
        print(x)
        return x

    def clamp(self, x):
        #print(x[0:50])
        m = x.max().item()
        peak = m*0.01
        threshold = m*0.85
        threshold = torch.full_like(x, threshold)
        selections = x.gt(threshold)
        x.fill_(0.)
        x[selections] = peak
        #print(x[0:50])
        return x



#
