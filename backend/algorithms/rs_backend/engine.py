"""base class for pool
The pool object will contain the models under optimization.
"""
from .analysis import Analysis

import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
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
            self.elite.copy_(self.vector)
            self.jumped = True

    def set_vector(self):
        if not self.jumped:
            self.vector.copy_(self.elite)

    def generate(self):
        self.vector.uniform_(-1., 1.)

    def update_weights(self, model):
        torch.nn.utils.vector_to_parameters(self.vector, model.parameters())


#
