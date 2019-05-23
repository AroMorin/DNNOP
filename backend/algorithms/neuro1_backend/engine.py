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
        self.mu = 0.000001

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
            self.reinforce()
        else:
            self.erode()
        for i in range(len(model.a1)):
            current_a = model.a1[i]
            model.fc1.weight.requires_grad_(False)
            model.fc1.weight.fill_(self.nu)
            model.fc1.weight[current_a, :].fill_(self.mu)
            model.fc1.weight.clamp_(0., 1.0)
            v = model.fc1.bias[:]
            v = self.bias(model.fc1.bias[:], current_a)
            model.fc1.bias[:] = v

            higher_a = model.a1[i]
            current_a = model.a2[i]
            model.fc2.weight.requires_grad_(False)
            model.fc2.weight.fill_(self.nu)
            model.fc2.weight[current_a, :].fill_(self.mu)
            v = model.fc2.bias[:]
            v = self.bias(v, current_a)
            model.fc2.bias[:] = v
            #print(model.fc2.weight[current_a])

            higher_a = model.a2[i]
            current_a = model.a3[i]
            model.fc3.weight.requires_grad_(False)
            model.fc3.weight.fill_(self.nu)
            model.fc3.weight[current_a, :].fill_(self.mu)
            v = model.fc3.bias[:]
            v = self.bias(v, current_a)
            model.fc3.bias[:] = v

            higher_a = model.a3[i]
            current_a = model.a4[i]
            model.fc4.weight.requires_grad_(False)
            model.fc4.weight.fill_(self.nu)
            model.fc4.weight[current_a, :].fill_(self.mu)
            v = model.fc4.bias[:]
            v = self.bias(v, current_a)
            model.fc4.bias[:] = v
        #print(model.a1[:])
        #print(model.fc1.weight[0])

    def reinforce(self):
        self.nu = 0.0
        self.mu = 0.9

    def erode(self):
        self.nu = 0.2
        self.mu = 0.0

    def get_v(self, v, higher_a):
        v.mul_(self.nu)
        p = v[:, higher_a]
        p.mul_(self.mu)
        v[:, higher_a] = p
        v.clamp_(0., 1.0)
        return v

    def bias(self, v, current_a):
        v.mul_(self.nu)
        p = v[current_a]
        p.add_(self.mu)
        v[current_a] = p
        v.clamp_(0., 1.0)
        return v

    def update_weights(self, params):
        torch.nn.utils.vector_to_parameters(self.vector, params)


#
