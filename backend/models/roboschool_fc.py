"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        ins = model_params['in features']
        outs = model_params['number of outputs']
        self.lmin = model_params['min action 1']
        self.lmax = model_params['max action 1']
        self.l = model_params['noise limit']
        self.fc1 = nn.Linear(ins, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, outs)
        self.drop = nn.Dropout(0.1)
        #self.act = nn.ReLU()
        self.act = nn.Tanh()

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params

    def generate_noise(self, x):
        n = torch.empty_like(x)
        n.normal_(mean=0., std=self.l)
        return n.cuda()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x.round_()
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc3(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc4(x)
        #noise = self.generate_noise(x)
        #x.add_(noise)
        #x.clamp_(self.lmin, self.lmax)
        return x
