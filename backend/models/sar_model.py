"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import canberra as distance

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        ins = model_params['in features']
        self.out_size = model_params['number of outputs']
        self.eta = model_params['noise limit']
        self.nu = 0.  # Noise parameter to action
        self.observations = []
        self.actions = []
        self.min_dist = 0.1
        self.in_table = False
        self.idx = 0
        self.x_0 = None
        self.state = 0
        self.peak = 0.1

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params

    def forward(self, x):
        self.reset_state()
        self.lookup(x)
        if self.in_table:  # Observation too similar to something in table
            action = self.actions[self.idx]
        else:  # Observation fairly new
            action = self.move()
        self.repeat(action)
        return self.val.cpu().detach().numpy()

    def reset_state(self):
        self.in_table = False
        if self.state>self.out_size-1:
            self.state = 0

    def lookup(self, x):
        #x = self.zero_out(x)
        print(len(self.observations))
        if len(self.observations)>0:
            d, closest = self.calculate_distance(x)
            print("Distance: ", d, self.idx)
            self.in_table = d<self.min_dist

    def zero_out(self, x):
        if self.x_0 is None:
            self.x_0 = x
            return x
        x1 = self.x_0 - x
        self.x_0 = x
        return x1

    def calculate_distance(self, x):
        min_d = float('inf')
        closest = 0
        for i, x2 in enumerate(self.observations):
            d = distance(x, x2)
            if d<min_d:  # Update min_d
                min_d = d
                closest = i
        self.idx = closest
        return min_d, closest

    def move(self):
        a = np.zeros((self.out_size,))
        idx = int(self.state)
        a[idx] = np.random.choice([-self.peak, self.peak], p=[0.5, 0.5])
        self.state+=0.25
        return a

    def get_random_action(self):
        a = np.random.normal(0, 0.3, (self.out_size,))
        action = np.clip(a, -1., 1.)
        zeros = np.zeros((self.out_size,))
        choice = np.random.choice([0., 1.], p=[0.3, 0.7])
        if choice == 0:
            return zeros
        else:
            print("taking random action------------")
            return action
