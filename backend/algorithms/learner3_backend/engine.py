"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .novelty import Novelty
from .weights import Weights
from .analysis import Analysis
from .integrity import Integrity
from .selection_p import Selection_P

import time
import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.elite = None
        self.analyzer = Analysis(hyper_params)
        self.weights = Weights(self.model.state_dict())
        self.ns = Novelty(hyper_params)
        self.noise = Noise(hyper_params, self.weights.vector)
        self.integrity = Integrity(hyper_params)
        self.selection_p = Selection_P(hyper_params, self.noise.vec_length)

    def set_elite(self):
        if self.analyzer.replace:
            self.elite = self.weights.vector.clone()

    def get_novelty(self, score):
        print(score)
        self.engine.ns.update(score)
        agg_score = score+self.engine.ns.value
        print(agg_score)
        return agg_score

    def update_state(self, improved):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.set_elite()
        self.selection_p.update_state(self.analyzer.replace, self.noise.choices)
        self.integrity.set_integrity(self.analyzer.improved)
        # Define noise vector
        self.noise.update_state(self.integrity.value, self.selection_p.p)

    def generate(self):
        new_vector = self.elite.clone()
        new_vector.add_(self.noise.vector)
        self.weights.update(new_vector)
        self.model.load_state_dict(self.weights.current)

#
