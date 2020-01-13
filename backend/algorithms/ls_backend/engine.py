"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis

import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
        self.noise = Noise(hyper_params, self.vector)
        self.analyzer = Analysis(hyper_params)
        self.elite = self.vector.clone()
        self.jumped = False

    def analyze(self, score, top_score):
        score = score.float()
        top_score = top_score.float()
        self.analyzer.analyze(score, top_score)
        #self.frustration.update(self.analyzer.improved)

    def set_elite(self):
        self.jumped = False
        if self.analyzer.replace:
            self.elite[self.noise.direction.value] = self.vector[self.noise.direction.value]
            #self.elite.clamp_(-0.9, 0.9)
            #self.elite.copy_(self.vector)
            self.jumped = True
            #self.frustration.reset_state()

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.noise.update_state(self.analyzer.replace)

    def set_vector(self):
        if not self.jumped:
            #self.vector.copy_(self.elite)
            elite_vals = self.elite[self.noise.direction.value]
            self.vector[self.noise.direction.value] = elite_vals

    def generate(self):
        noise_vals = self.vector[self.noise.direction.value]+self.noise.magnitude
        self.vector[self.noise.direction.value] = noise_vals
        #self.vector.add_(self.noise.vector)

    def update_weights(self, model):
        torch.nn.utils.vector_to_parameters(self.vector, model.parameters())


#
