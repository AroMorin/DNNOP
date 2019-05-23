"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch
import random

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        self.fc1 = nn.Linear(model_params['in features'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, model_params['number of outputs'])
        self.act = nn.ReLU()
        self.a1 = []
        self.a2 = []
        self.a3 = []
        self.a4 = []
        self.step = 0
        self.max = -float('inf')
        self.min = float('inf')
        self.prev = None
        self.peak = 0.05
        self.threshold = 1.5
        self.mu = 0.01
        self.excitation = None
        self.dropout = 0.3

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = self.zero_out(x)
        #self.update(x)
        #self.set_limits(x)
        #x = self.normalize(x)
        #x = x.sub_(x.mean())
        #self.step_()

        x = self.fc1(x)
        x = self.act(x)
        #x = self.fire(x)
        self.set_a1(x)

        x = self.fc2(x)
        x = self.act(x)
        #x = self.fire(x)
        self.set_a2(x)

        x = self.fc3(x)
        x = self.act(x)
        #x = self.fire(x)
        self.set_a3(x)

        x = self.fc4(x)
        print(x)
        x1 = self.act(x)
        #x1 = self.fire(x)
        self.set_a4(x1)
        return x

    def set_limits(self, x):
        ma = x.max()
        mi = x.min()
        if ma > self.max:
            self.max = ma
        if mi < self.min:
            self.min = mi

    def zero_out(self, x):
        if self.step == 0:
            self.prev = x.clone()
            return x
        x1 = x.sub(self.prev)
        if not self.step == 0:
            self.prev = x.clone()
        return x1

    def step_(self):
        self.step += 1

    def normalize(self, x):
        x = x.sub(self.min)
        d = self.max-self.min
        n = x.div(d)
        return n

    def filter(self, x):
        ma = x.max().item()
        mi = x.min().item()
        f = 0.85
        t1 = f*ma
        t2 = f*mi
        t1 = torch.full_like(x, t1)
        t2 = torch.full_like(x, t2)
        select = x.gt(t1)
        select2 = x.lt(t2)
        x.fill_(0.)
        x[select] = self.peak
        x[select2] = self.peak
        return x

    def fire(self, x):
        m = x.max().item()
        t = 0.85*m
        threshold = torch.full_like(x, t)
        saturated = x.gt(threshold)
        x.fill_(0.)
        x[saturated] = self.peak
        num_drop = int(self.dropout*x.size()[0])
        blackout = torch.randint(0, x.size()[0], (num_drop,))
        val = random.choice([0., self.peak])
        x[blackout] = val
        return x

    def set_a1(self, x):
        active = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a1.append(i[active])

    def set_a2(self, x):
        active = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a2.append(i[active])

    def set_a3(self, x):
        active = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a3.append(i[active])

    def set_a4(self, x):
        active = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a4.append(i[active])

    def excite(self, x):
        y = x.clone()
        if self.step != 0:
            self.excitation.add_(y)
        else:
            self.excitation = x
        return self.excitation

    def update(self, x):
        dead = x.eq(0.)
        active = x.ne(0.)
        i = torch.arange(x.size()[0])
        dead = i[dead]
        active = i[active]

        v = self.fc1.weight[:, dead]
        v.sub_(0.001)
        v.clamp_(0., 1.0)
        self.fc1.weight[:, dead] = v

        v = self.fc1.weight[:, active]
        v.add_(0.1)
        v.clamp_(0., 1.0)
        self.fc1.weight[:, active] = v
        print(self.fc1.weight[0])

        #v = self.fc2.weight[self.a2, :]
        #v.sub_(self.mu)
        #v[:, self.a1].add_(2*self.mu)
        #v.clamp_(0., 1.0)
        #self.fc2.weight[self.a2, :] = v

        #v = self.fc3.weight[self.a3, :]
        #v.sub_(self.mu)
        #v[:, self.a2].add_(2*self.mu)
        #v.clamp_(0., 1.0)
        #self.fc3.weight[self.a3, :] = v

        #v = self.fc4.weight[self.a4, :]
        #v.sub_(self.mu)
        #v[:, self.a3].add_(2*self.mu)
        #v.clamp_(0., 1.0)
        #self.fc4.weight[self.a4, :] = v




#
