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
        self.fc1 = nn.Linear(ins, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, outs)
        #self.fc5 = nn.Linear(256, ins)
        self.dropout = nn.Dropout(0.9)
        self.ex0 = []
        self.ex1 = []
        self.ex2 = []
        self.ex3 = []
        #self.ex5 = []
        self.ap1 = torch.zeros(self.fc1.weight.data.size()[0]).half().cuda()
        self.ap2 = torch.zeros(self.fc2.weight.data.size()[0]).half().cuda()
        self.ap3 = torch.zeros(self.fc3.weight.data.size()[0]).half().cuda()
        #self.ap5 = torch.zeros(self.fc5.weight.data.size()[0])
        self.x_0 = None  # Previous observation
        self.peak = 0.5
        self.ap_t = 3.  # Action potential threshold
        self.increment = 1
        self.reps = 3
        self.rep = 0
        self.val = torch.zeros(outs).half().cuda()

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
        noise.normal_(0, 0.5)
        noise = self.dropout(noise)
        x.add_(noise)
        x = self.zero_out(x)
        x = x.half().squeeze()
        self.set_ex0(x)


        x = self.fc1(x)
        # AP should be set after weights not before (to kill neurons and excitations)
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
        action = self.fc4(x).squeeze().clamp_(-1., 1.)
        action = action.cpu().detach().numpy()
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
        self.ap1[up] = self.ap1[up]+x[up]
        down = x.lt(0)
        self.ap1[down] = self.ap1[down]+x[down]

    def noise(self, x, up, down):
        indices = np.arange(x.size()[0])
        others = np.delete(indices, up.cpu().numpy())
        others = np.delete(others, down.cpu().numpy())
        others = others.tolist()
        rand = np.random.choice(others, int(0.02*len(others)), replace=False)
        mid = int(len(rand.tolist())/2)
        up_noise = rand[:mid]
        down_noise = rand[mid:]
        return up_noise, down_noise

    def set_ap2(self, x):
        up = x.gt(0)
        self.ap2[up] = self.ap2[up]+x[up]
        down = x.lt(0)
        self.ap2[down] = self.ap2[down]+x[down]

    def set_ap3(self, x):
        up = x.gt(0)
        self.ap3[up] = self.ap3[up]+x[up]
        down = x.lt(0)
        self.ap3[down] = self.ap3[down]+x[down]

    def fire_fc1(self, x):
        sat_up = self.ap1.gt(self.ap_t)
        sat_down = self.ap1.lt(-self.ap_t)
        self.ap1[sat_up] = 0
        self.ap1[sat_down] = 0
        x = self.fire(x, sat_up, sat_down)
        return x

    def fire(self, x, sat_up, sat_down):
        i = torch.arange(x.size()[0])
        sat_up = i[sat_up].numpy()
        sat_down = i[sat_down].numpy()
        x[:] = 0.
        x[sat_up] = self.peak
        x[sat_down] = -self.peak
        return x

    def fire_fc2(self, x):
        sat_up = self.ap2.gt(self.ap_t)
        sat_down = self.ap2.lt(-self.ap_t)
        self.ap2[sat_up] = 0
        self.ap2[sat_down] = 0
        x = self.fire(x, sat_up, sat_down)
        return x

    def fire_fc3(self, x):
        sat_up = self.ap3.gt(self.ap_t)
        sat_down = self.ap3.lt(-self.ap_t)
        self.ap3[sat_up] = 0
        self.ap3[sat_down] = 0
        x = self.fire(x, sat_up, sat_down)
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

    def measure(self, x):
        active = x.ne(0.)
        i = torch.arange(x.size()[0])
        return i[active]

    def repeat(self, x):
        if not x.equal(self.val):
            if self.rep > self.reps:
                self.val = x.clone()
                self.rep=0
            else:
                self.rep +=1
        else:
            self.rep +=1
        print(self.val, self.rep)
#
