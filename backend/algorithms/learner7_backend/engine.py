"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
from .selection_p import Selection_P
from .frustration import Frustration

import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.analyzer = Analysis(hyper_params)
        self.frustration = Frustration(hyper_params)
        self.integrity = Integrity(hyper_params)
        self.vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.elite = self.vector
        self.noise = Noise(hyper_params, self.vector)
        self.selection_p = Selection_P(hyper_params, self.noise.vec_length)
        self.jumped = False

    def analyze(self, score, top_score):
        self.analyzer.analyze(score, top_score)
        #self.frustration.update(score, top_score)

    def set_elite(self):
        self.jumped = False
        if self.analyzer.replace or self.frustration.jump:
            self.elite = self.vector.clone()
            self.jumped = True

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.selection_p.update_state(self.elite)
        self.integrity.set_integrity(self.analyzer.improved)
        # Define noise vector
        limits = (self.elite.min(), self.elite.max())
        self.noise.update_state(self.integrity.value, self.selection_p.p, limits)

    def generate(self):
        new_vector = self.elite.clone()
        new_vector.add_(self.noise.vector)
        #new_vector.clamp_(-0.3, 0.3)
        self.vector = new_vector

    def update_weights(self):
        torch.nn.utils.vector_to_parameters(self.vector, self.model.parameters())


#
