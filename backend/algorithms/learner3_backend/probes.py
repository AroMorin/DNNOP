"""Base class for probes"""

from .perturbation import Perturbation
import torch

class Probes(object):
    def __init__(self, hp):
        self.vector = None

    def generate(self, vector, perturb):
        """Set the new probes based on the calculated anchors."""
        probe = vector.clone()
        perturb.apply(probe)
        self.vector = probe


#
