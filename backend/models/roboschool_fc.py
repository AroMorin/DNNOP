"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        ins = model_params['in features']
        outs = model_params['number of outputs']
        self.fc1 = nn.Linear(ins, 16)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(16, outs)
        self.drop = nn.Dropout(0.0)
        self.act = nn.ReLU()
        #self.act = nn.Tanh()

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
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        #x = self.fc2(x)
        #x = self.act(x)
        #x = self.drop(x)
        #x = self.fc3(x)
        #x = self.act(x)
        #x = self.drop(x)
        x = self.fc4(x)
        return x
