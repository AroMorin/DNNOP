"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import numpy as np
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        ins = model_params['in features']
        outs = model_params['number of outputs']
        self.out_size = outs
        self.fc1 = nn.Linear(ins, 512)
        self.fc2 = nn.Linear(512, 16)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(16, outs)
        self.drop = nn.Dropout(0.1)
        self.act = nn.ReLU()
        #self.act = nn.Tanh()
        self.reps = 20
        self.rep = 0
        self.step = 0
        self.val = torch.zeros(outs).half().cuda()

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
        n.normal_(mean=0., std=0.3)
        return n.cuda()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        #x = self.drop(x)
        #x = self.fc3(x)
        #x = self.act(x)
        #x = self.drop(x)
        x = self.fc4(x).squeeze().clamp_(-1., 1.)
        #self.repeat(x)
        return x.cpu().detach().numpy()

    def repeat(self, x):
        if self.rep > self.reps:
            self.reset(x)
            self.rep=0
        else:
            self.rep +=1
        print(self.val, self.rep)

    def reset(self, x):
        default = torch.zeros(self.out_size).cuda()
        choice = np.random.choice([0, 1], p=[0.5, 0.5])
        if choice == 0:
            self.val = default
        else:
            self.val = x.clone()
