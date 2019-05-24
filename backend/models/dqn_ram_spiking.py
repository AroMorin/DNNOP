"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        self.fc1 = nn.Linear(model_params['in features'], 256)
        self.norm = nn.LayerNorm(256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, model_params['number of outputs'])
        self.max = 0

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
        x = self.fc1(x)
        x = self.clamp(x)
        x = self.fc2(x)
        x = self.clamp(x)
        x = self.fc3(x)
        x = self.clamp(x)
        #print(x.nonzero().shape[0])
        x = self.fc4(x)
        #print(x)
        #exit()
        return x

    def normalize(self, x):
        n = x.div(self.max)
        return n

    def set_max(self, x):
        self.max = x.max()

    def clamp(self, x):
        m = x.max().item()
        peak = m*0.01
        threshold = m*0.85
        threshold = torch.full_like(x, threshold)
        selections = x.gt(threshold)
        x.fill_(0.)
        x[selections] = peak
        return x
