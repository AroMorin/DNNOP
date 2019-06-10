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
        self.kappa = 0.01

    def analyze(self, score, top_score):
        self.analyzer.analyze(score, top_score)
        self.frustration.update(self.analyzer.replace)

    def set_elite(self):
        self.jumped = False
        if self.analyzer.replace or self.frustration.jump:
            self.elite = self.vector.clone()
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
        self.vector = self.elite.clone()
        self.update_v(model)

    def reinforce(self):
        self.mu = 0.05

    def erode(self):
        self.mu = -0.1

    def update_v(self, model):
        v = model.fc1.weight[:, :]
        v.sub_(0.002)
        v.clamp_(0., 1.0)
        model.fc1.weight[:, :] = v
        v = model.fc1.weight[:, model.ex1]
        v.add_(self.mu)
        v.clamp_(0., 1.0)
        model.fc1.weight[:, model.ex1] = v
        #print(model.fc1.weight.data[0:30])

        v = model.fc2.weight[:, :]
        v.sub_(0.002)
        v.clamp_(0., 1.0)
        model.fc2.weight[:, :] = v
        v = model.fc2.weight[:, model.ex2]
        v.add_(self.mu)
        v.clamp_(0., 1.0)
        model.fc2.weight[:, model.ex2] = v

        v = model.fc3.weight[:, :]
        v.sub_(0.002)
        v.clamp_(0., 1.0)
        model.fc3.weight[:, :] = v
        v = model.fc3.weight[:, model.ex3]
        v.add_(self.mu)
        v.clamp_(0., 1.0)
        model.fc3.weight[:, model.ex3] = v


#
