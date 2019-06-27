"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
#from .frustration import Frustration

import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.vector = torch.nn.utils.parameters_to_vector(model.parameters())
        self.noise = Noise(hyper_params, self.vector)
        self.analyzer = Analysis(hyper_params)
        self.integrity = Integrity(hyper_params)
        #self.frustration = Frustration(hyper_params)
        self.elite = torch.empty_like(self.vector)
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
            #self.frustration.reset_state()

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        #self.selection_p.update_state(self.analyzer.replace, self.noise.choices)
        self.integrity.set_integrity(self.analyzer.improved)

    def generate(self):
        self.vector.copy_(self.elite)
        self.create_noise()
        self.vector.add_(self.noise.vector)

    def create_noise(self):
        # Define noise vector
        self.noise.update_state(self.integrity.value, self.analyzer.replace)

    def update_weights(self, model):
        torch.nn.utils.vector_to_parameters(self.vector, model.parameters())


#
