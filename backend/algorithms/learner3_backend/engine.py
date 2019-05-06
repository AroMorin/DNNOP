"""base class for pool
The pool object will contain the models under optimization.
"""
from .elite import Elite
from .noise import Noise
from .memory import Memory
from .weights import Weights
from .integrity import Integrity
from .selection_p import Selection_P

import time
import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.elite = Elite(hyper_params)
        self.weights = Weights(self.model.state_dict())
        self.noise = Noise(hyper_params, self.weights.vector)
        self.integrity = Integrity(hyper_params)
        self.selection_p = Selection_P(hyper_params, self.noise.vec_length)

    def analyze(self, score):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.selection_p.update_state(score, self.noise.choices)
        self.integrity.set_integrity(score)
        # Define noise vector
        self.noise.update_state(self.integrity.value, self.selection_p.p)

    def generate(self):
        new_vector = self.elite.vector.clone()
        new_vector.add_(self.noise.vector)
        self.weights.update(new_vector)
        self.model.load_state_dict(self.weights.current)

#
