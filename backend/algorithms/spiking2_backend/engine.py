"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
from .frustration import Frustration

import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.analyzer = Analysis(hyper_params)
        self.frustration = Frustration(hyper_params)
        self.integrity = Integrity(hyper_params)
        for param in self.model.parameters():
            print(param)
            print(param.size())
        exit()

        self.vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.elite = self.vector
        self.noise = Noise(hyper_params, self.vector)
        self.jumped = False

    def analyze(self, score, top_score):
        self.analyzer.analyze(score, top_score)
        self.frustration.update(score, top_score)

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

    def generate(self):
        new_vector = self.elite.clone()
        print(new_vector[0:19])
        new_vector.add_(self.noise.vector)
        new_vector.clamp_(0.01, 0.99)
        #new_vector[self.noise.choices] = self.noise.vector
        self.vector = new_vector

    def update(self, model):
        if self.analyzer.replace:
            self.reinforce()
        else:
            self.erode()
        for i in range(len(model.a1)):
            current_a = model.a1[i]
            #print(current_a)
            v = model.fc1.weight[:]
            v.mul_(self.nu)
            p = v[current_a, :]
            p.mul_(self.mu)
            v[current_a, :] = p
            v.clamp_(0., 1.0)
            model.fc1.weight[:] = v
            v = model.fc1.bias[:]
            v = self.bias(v, current_a)
            model.fc1.bias[:] = v

            higher_a = model.a1[i]
            current_a = model.a2[i]
            v = model.fc2.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc2.weight[current_a, :] = v
            v = model.fc2.bias[:]
            v = self.bias(v, current_a)
            model.fc2.bias[:] = v
            #print(model.fc2.weight[current_a])

            higher_a = model.a2[i]
            current_a = model.a3[i]
            v = model.fc3.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc3.weight[current_a, :] = v
            v = model.fc3.bias[:]
            v = self.bias(v, current_a)
            model.fc3.bias[:] = v

            higher_a = model.a3[i]
            current_a = model.a4[i]
            v = model.fc4.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc4.weight[current_a, :] = v
            v = model.fc4.bias[:]
            v = self.bias(v, current_a)
            model.fc4.bias[:] = v
        #print(model.a1[:])
        #print(model.fc1.weight[0])

    def reinforce(self):
        self.mu = 1.0002
        self.nu = 0.9995

    def erode(self):
        self.mu = 0.9995
        self.nu = 1.0002

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


    def update_weights(self):
        torch.nn.utils.vector_to_parameters(self.vector, self.model.parameters())


#
