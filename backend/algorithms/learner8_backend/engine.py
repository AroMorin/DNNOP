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
            #current_a = model.a1[i]
            #v = model.fc1.weight[current_a, :]
            #v = v.add(self.mu)
            #v = v.clamp(0., 1.0)
            #model.fc1.weight[current_a, :] = v

            higher_a = model.a1[i]
            current_a = model.a2[i]
            v = model.fc2.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc2.weight[current_a, :] = v
            #print(model.fc2.weight[current_a])

            higher_a = model.a2[i]
            current_a = model.a3[i]
            v = model.fc3.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc3.weight[current_a, :] = v

            higher_a = model.a3[i]
            current_a = model.a4[i]
            v = model.fc4.weight[current_a, :]
            v = self.get_v(v, higher_a)
            model.fc4.weight[current_a, :] = v
        #print(model.a1[:])
        #print(model.fc1.weight[0])

    def reinforce(self):
        self.mu = 0.0001

    def erode(self):
        self.mu = 0.00001

    def get_v(self, v, higher_a):
        v.sub_(self.mu)
        p = v[:, higher_a]
        p.add_(self.mu*2.)
        v[:, higher_a] = p
        v.clamp_(0., 1.0)
        return v

    def update_weights(self, params):
        torch.nn.utils.vector_to_parameters(self.vector, params)


#
