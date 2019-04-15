"""Base class for probes"""

from .perturbation import Perturbation
import torch

class Memory(object):
    def __init__(self, hp):
        self.vector = None
        self.observations = []
        self.inferences = []
        self.scores = []
        self.matrix = None

    def update(self, observation, inference, score):
        if observation not in self.observations:
            self.observations.append(observation)
            self.inferences.append(inference)
            self.scores.append(score)

    def evaluate_hypothesis(self, hypothesis):
        pass

    def generate(self, vector, perturb):
        """Set the new probes based on the calculated anchors."""
        print(vector[0:10])
        probe = vector.clone()
        perturb.apply(probe)
        self.vector = probe
        print(self.vector[0:10])


#
