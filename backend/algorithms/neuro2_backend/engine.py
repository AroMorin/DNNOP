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
        if self.analyzer.replace:
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
        vec = self.elite.clone()
        #print(vec[0:100])
        self.update_v(vec)
        vec.add_(self.noise.vector)
        self.vector = vec
        torch.nn.utils.vector_to_parameters(self.vector, model.parameters())

    def reinforce(self):
        self.mu = 0.002
        self.nu = -0.001

    def erode(self):
        self.mu = -0.002
        self.nu = 0.001

    def update_v(self, v):
        ma = v.max().item()
        mi = v.min().item()
        strongest = v.gt(0.3*ma).nonzero()
        if mi<0.:
            weakest = v.lt(0.3*mi).nonzero()
        else:
            weakest = v.lt(1.7*mi).nonzero()
        v[strongest] = v[strongest].add(self.mu)
        v[weakest] = v[weakest].add(self.nu)

    def update_weights__(self, params):
        torch.nn.utils.vector_to_parameters(self.vector, params)


#
