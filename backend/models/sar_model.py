"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

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

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in features": 128,
                            "number of outputs": 18
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params

    def forward(self, x):
        d, closest = self.calculate_distance(x)
        if d<self.min_dist:
            self.observations.extend(x)
            action = self.get_random_action()
        else:
            action = self.actions[closest]
        return action

    def get_random_action(self):
        return np.random.uniform()
