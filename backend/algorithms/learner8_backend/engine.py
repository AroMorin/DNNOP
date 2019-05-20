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
        self.mu = 0.1

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

    def update(self, model):
        if self.analyzer.replace:
            self.reinforce(model)
        else:
            self.erode(model)
        print(model.a1)
        print(model.a2)
        print(model.a3)
        print(model.a4)
        print(model.fc1.weight.size())
        print(model.fc2.weight.size())
        print(model.fc3.weight.size())
        print(model.fc4.weight.size())

        v = model.fc1.weight[model.a1, :]
        v.add_(self.mu)
        model.fc1.weight[model.a1, :] = v

        v = model.fc2.weight[model.a2, :]
        v[:, model.a1].add_(self.mu)
        model.fc2.weight[model.a2, :] = v

        v = model.fc3.weight[model.a3, :]
        v[:, model.a2].add_(self.mu)
        model.fc3.weight[model.a3, :] = v

        v = model.fc4.weight[model.a4, :]
        v[:, model.a3].add_(self.mu)
        model.fc4.weight[model.a4, :] = v

    def reinforce(self, model):
        self.mu = 0.01

    def erode(self, model):
        self.mu = -0.01

    def update_weights(self, params):
        torch.nn.utils.vector_to_parameters(self.vector, params)


#
