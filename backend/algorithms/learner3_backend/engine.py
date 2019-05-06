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
        self.mem = Memory(hyper_params)
        self.elite = Elite(hyper_params)
        self.noise = Noise(hyper_params, self.weights.vector)
        self.weights = Weights(self.model.state_dict())
        self.integrity = Integrity(hyper_params)
        self.selection_p = Selection_P(hp, self.noise.vec_length)

    def analyze(self, observation, inference, score):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        #self.mem.update_state(observation, label, inference, score)
        self.selection_p.update_state(score, self.noise.choices)
        self.integrity.set_integrity(score)
        # Define noise magnitude and scale
        self.noise.update_state(self.integrity.value, self.selection_p.p)

    def generate(self):
        new_vector = self.elite.vector.clone()
        new_vector.add_(self.noise.vector)
        self.weights.update(new_vector)
        self.model.load_state_dict(self.weights.current)

    def evaluate(self):
        self.mem.evaluate_model(self.model)
        if not self.mem.desirable:
            self.analyzer.suspend_reality()
            self.perturb.suspend_reality()
        for i in range(100):
            if not self.mem.desirable:
                eval = torch.tensor(self.mem.eval, device='cuda', dtype=torch.float)
                self.analyzer.analyze(eval)
                #self.perturb.update_state(self.analyzer)
                self.generate()
                self.mem.evaluate_model(self.model)
            else:
                print("Expected: %f------------------------" %self.mem.eval)
                return
        self.analyzer.restore_reality()
        self.perturb.restore_reality()


#
