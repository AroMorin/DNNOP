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
        #self.fc5 = nn.Linear(256, ins)
        self.dropout = nn.Dropout(0.9)
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
        self.peak = 0.05
        self.ap_t = 2  # Action potential threshold
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
        noise = torch.empty_like(x)
        noise.normal_(0, 0.01)
        noise = self.dropout(noise)
        #x.add_(noise)
        x = self.zero_out(x)
        x = x.half().squeeze()
        self.set_ex0(x)
        x = self.fc1(x)

        self.set_ap1(x)
        x = self.fire_fc1(x)
        self.set_ex1(x)

        x = self.fc2(x)
        self.set_ap2(x)
        x = self.fire_fc2(x)
        self.set_ex2(x)

        x = self.fc3(x)
        self.set_ap3(x)
        x = self.fire_fc3(x)
        self.set_ex3(x)

        action = self.fc4(x).squeeze()
        return action

    def zero_out(self, x):
        if self.x_0 is None:
            self.x_0 = x.clone()
            return x
        x1 = x.sub(self.x_0)  # In-place is not possible
        self.x_0 = x.clone()
        return x1

    def set_ap1(self, x):
        up = x.gt(0)
        self.ap1[up] = self.ap1[up].add(self.increment)
        down = x.lt(0)
        self.ap1[down] = self.ap1[down].sub(self.increment)

    def set_ap2(self, x):
        up = x.gt(0)
        self.ap2[up] = self.ap2[up].add(self.increment)
        down = x.lt(0)
        self.ap2[down] = self.ap2[down].sub(self.increment)

    def set_ap3(self, x):
        up = x.gt(0)
        self.ap3[up] = self.ap3[up].add(self.increment)
        down = x.lt(0)
        self.ap3[down] = self.ap3[down].sub(self.increment)

    def set_ap4(self, x):
        up = x.gt(0)
        self.ap4[up] = self.ap4[up].add(self.increment)
        down = x.lt(0)
        self.ap4[down] = self.ap4[down].sub(self.increment)

    def fire_fc1(self, x):
        # Regular firing
        sat_up = self.ap1.gt(self.ap_t)
        sat_down = self.ap1.lt(-self.ap_t)
        x = self.fire(x, sat_up, sat_down)
        self.ap1[sat_up] = 0
        self.ap1[sat_down] = 0
        # Noise firing
        #i = torch.arange(x.size()[1])
        return x

    def fire_fc2(self, x):
        sat_up = self.ap2.gt(self.ap_t)
        sat_down = self.ap2.lt(-self.ap_t)
        x = self.fire(x, sat_up, sat_down)
        self.ap2[sat_up] = 0
        self.ap2[sat_down] = 0
        return x

    def fire_fc3(self, x):
        sat_up = self.ap3.gt(self.ap_t)
        sat_down = self.ap3.lt(-self.ap_t)
        x = self.fire(x, sat_up, sat_down)
        self.ap3[sat_up] = 0
        self.ap3[sat_down] = 0
        return x

    def fire_fc4(self, x):
        sat_up = self.ap4.gt(self.ap_t)
        sat_down = self.ap4.lt(-self.ap_t)
        x = self.fire(x, sat_up, sat_down)
        self.ap4[sat_up] = 0
        self.ap4[sat_down] = 0
        return x

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

    def fire(self, x, sat_up, sat_down):
        i = torch.arange(x.size()[0])
        sat_up = i[sat_up].numpy()
        sat_down = i[sat_down].numpy()
        x[sat_up] = x[sat_up].tanh().mul(self.peak)
        x[sat_down] = x[sat_down].tanh().mul(self.peak)
        indices = np.arange(x.size()[0])
        others = np.delete(indices, sat_up)
        others = np.delete(others, sat_down)
        others = others.tolist()
        x[others] = x[others].fill_(0.)
        rand = np.random.choice(others, int(0.05*len(others)), replace=False)
        x[rand].fill_(0.1)
        return x

    def measure(self, x):
        active = x.ne(0.)
        i = torch.arange(x.size()[0])
        return i[active]

#
