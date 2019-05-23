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
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.step = 0
        self.max = -float('inf')
        self.min = float('inf')
        self.prev = None

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
        #self.update(x)
        #self.set_limits(x)
        #x = self.normalize(x)
        #x = x.sub_(x.mean())
        #self.step_()

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc4(x)
        #x = self.act(x)
        #print(x.argmax())
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



#
