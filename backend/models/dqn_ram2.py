"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        self.fc1 = nn.Linear(model_params['in features'], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, model_params['number of outputs'])
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.a4 = None
        self.a1 = None
        self.step = 0
        self.max = -float('inf')
        self.min = float('inf')
        self.prev = None
        self.peak = 0.05
        self.threshold = 0.05

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
        x = self.fc1(x)
        x = self.clamp(x)
        self.set_a1(x)

        x = self.fc2(x)
        x = self.clamp(x)
        self.set_a2(x)

        x = self.fc3(x)
        x = self.clamp(x)
        self.set_a3(x)

        x = self.fc4(x)
        x1 = self.clamp(x)
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
            self.prev = x
            return x
        x1 = x.sub(self.prev)
        if not self.step == 0:
            self.prev = x
        return x1

    def step_(self):
        self.step += 1

    def normalize(self, x):
        x = x.sub(self.min)
        d = self.max-self.min
        n = x.div(d)
        return n

    def filter(self, x):
        threshold = 5.
        threshold = torch.full_like(x, threshold)
        selections = x.gt(threshold)
        x.fill_(0.)
        x[selections] = peak
        return x

    def clamp(self, x):
        threshold = torch.full_like(x, self.threshold)
        saturated = x.gt(threshold)
        x.fill_(0.)
        x[saturated] = self.peak
        return x

    def set_a1(self, x):
        indices = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a1 = i[indices]

    def set_a2(self, x):
        indices = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a2 = i[indices]

    def set_a3(self, x):
        indices = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a3 = i[indices]

    def set_a4(self, x):
        indices = x.eq(self.peak)
        i = torch.arange(x.size()[0])
        self.a4 = i[indices]





#
