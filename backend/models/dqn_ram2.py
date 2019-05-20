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
        self.step = 0
        self.max = 0
        self.prime = None
        self.pre = None
        self.post = None

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18,
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        self.set_max(x)
        x = self.zero_out(x)
        self.step_()
        x = self.normalize(x)
        x = self.fc1(x)
        x = self.clamp(x)
        self.set_pre(x)
        x = self.fc2(x)
        x = self.clamp(x)
        self.set_post(x)
        x = self.fc3(x)
        x = self.clamp(x)
        x = self.fc4(x)
        return x

    def set_max(self, x):
        self.max = x.max()

    def zero_out(self, x):
        if self.step == 0:
            self.prime = x
            return x
        x1 = x.sub(self.prime)
        if not self.step == 0:
            self.prime = x
        return x1

    def step_(self):
        self.step += 1

    def normalize(self, x):
        n = x.div(self.max)
        return n

    def clamp(self, x):
        peak = 0.05
        threshold = 0.05
        threshold = torch.full_like(x, threshold)
        selections = x.gt(threshold)
        x.fill_(0.)
        x[selections] = peak
        return x

    def set_pre(self, x):
        indices = x.eq(0.05)
        i = torch.arange(x.size()[0])
        self.pre = i[indices]

    def set_post(self, x):
        indices = x.eq(0.05)
        i = torch.arange(x.size()[0])
        self.post = i[indices]


#
