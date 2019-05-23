"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
from .frustration import Frustration

import torch

class Engine(object):
    def __init__(self, params, hyper_params):
        self.analyzer = Analysis(hyper_params)
        self.frustration = Frustration(hyper_params)
        self.integrity = Integrity(hyper_params)
        self.vector = torch.nn.utils.parameters_to_vector(params)
        self.elite = self.vector
        self.noise = Noise(hyper_params, self.vector)
        self.jumped = False

    def analyze(self, score, top_score):
        self.analyzer.analyze(score, top_score)
        self.frustration.update(score, top_score)

    def set_elite(self):
        self.jumped = False
        if self.analyzer.replace or self.frustration.jump:
            #self.elite = self.vector.clone()
            self.jumped = True

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.integrity.set_integrity(self.analyzer.improved)
        # Define noise vector
        self.noise.update_state(self.integrity.value)

    def update_weights(self, model):
        if self.analyzer.replace:
            self.reinforce()
        else:
            self.erode()
        self.update_w(model.fc1.weight)
        self.update_w(model.fc2.weight)
        self.update_w(model.fc3.weight)
        self.update_w(model.fc4.weight)
        self.update_b(model.fc1.bias)
        self.update_b(model.fc2.bias)
        self.update_b(model.fc3.bias)
        self.update_b(model.fc4.bias)
        print(model.fc1.weight[0])

    def reinforce(self):
        self.mu = 0.01
        self.nu = -0.01

    def erode(self):
        self.mu = -0.01
        self.nu = 0.01

    def update_w(self, v):
        v.requires_grad_(False)
        ma = v.max().item()
        mi = v.min().item()
        strongest = v.gt(0.5*ma).nonzero()
        weakest = v.lt(1.5*mi).nonzero()
        p = v[strongest[:, 0], strongest[:, 1]]
        p.add_(self.mu)
        v[strongest[:, 0], strongest[:, 1]] = p
        p = v[weakest[:, 0], weakest[:, 1]]
        p.add_(self.nu)
        v[weakest[:, 0], weakest[:, 1]] = p

    def update_b(self, v):
        v.requires_grad_(False)
        ma = v.max().item()
        mi = v.min().item()
        strongest = v.gt(0.5*ma).nonzero()
        weakest = v.lt(1.5*mi).nonzero()
        p = v[strongest]
        p.add_(self.mu)
        v[strongest] = p
        p = v[weakest]
        p.add_(self.nu)
        v[weakest] = p

    def update_weights__(self, params):
        torch.nn.utils.vector_to_parameters(self.vector, params)


#
