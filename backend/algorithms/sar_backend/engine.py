"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
from .frustration import Frustration
from .weights import Weights

import torch

class Engine(object):
    def __init__(self, params, hyper_params):
        self.analyzer = Analysis(hyper_params)
        self.frustration = Frustration(hyper_params)
        self.integrity = Integrity(hyper_params)
        self.vector = torch.nn.utils.parameters_to_vector(params)
        self.elite = self.vector
        self.noise = Noise(hyper_params, self.vector)
        self.weights = Weights(hyper_params, greed=True)
        self.jumped = False
        self.kappa = 0.01

    def analyze(self, feedback, top_score):
        self.analyzer.analyze(feedback, top_score)
        self.frustration.update(self.analyzer.analysis)

    def set_elite(self):
        self.jumped = False
        if self.analyzer.analysis == 'better' or self.frustration.jump:
            self.jumped = True

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.integrity.set_integrity(self.analyzer.improved)
        # Define noise vector
        self.noise.update_state(self.integrity.value)

    def update_weights(self, model):
        self.weights.update(self.analyzer.analysis, model)

#
