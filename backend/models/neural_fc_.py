"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch
import random
import numpy as np

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        ins = model_params['in features']
        outs = model_params['number of outputs']
        self.fc1 = nn.Linear(ins, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, outs)
        self.act = nn.Tanh()
        #self.fc5 = nn.Linear(256, ins)
        self.dropout = nn.Dropout(0.8)
        self.ex0 = []
        self.ex1 = []
        self.ex2 = []
        self.ex3 = []
        self.ex4 = []
        #self.ex5 = []
        self.ap1 = torch.zeros(self.fc1.weight.data.size()[0])
        self.ap2 = torch.zeros(self.fc2.weight.data.size()[0])
        self.ap3 = torch.zeros(self.fc3.weight.data.size()[0])
        self.ap4 = torch.zeros(self.fc4.weight.data.size()[0])
        #self.ap5 = torch.zeros(self.fc5.weight.data.size()[0])
        self.x_0 = None  # Previous observation
        self.peak = 0.1
        self.ap_t = 1  # Action potential threshold
        self.increment = 1
        #self.prediction = None

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params

    def forward(self, x):
        #x = self.zero_out(x)
        x = x.half().squeeze()
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        print(x)
        x = self.fc4(x)
        print(x)
        action = self.act(x)
        print(action)
        return action.squeeze()

    def add_noise(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, 0.2)
        noise = self.dropout(noise)
        x = x.add(noise)
        return x

    def zero_out(self, x):
        if self.x_0 is None:
            self.x_0 = x.clone()
            return x
        x1 = x.sub(self.x_0)  # In-place is not possible
        self.x_0 = x.clone()
        return x1

    def set_ex0(self, x):
        excitation = self.measure(x)
        self.ex0 = excitation

    def set_ex1(self, x):
        excitation = self.measure(x)
        self.ex1 = excitation

    def set_ex2(self, x):
        excitation = self.measure(x)
        self.ex2 = excitation

    def set_ex3(self, x):
        excitation = self.measure(x)
        self.ex3 = excitation

    def set_ex4(self, x):
        excitation = self.measure(x)
        self.ex4 = excitation

    def measure(self, x):
        active = x.ne(0.)
        i = torch.arange(x.size()[0])
        return i[active]

#
