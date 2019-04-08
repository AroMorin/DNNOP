"""Base class for probes"""

from .perturbation import Perturbation
import torch

class Probes(object):
    def __init__(self, hp):
        self.nb_probes = hp.nb_probes
        self.vectors = []
        self.probes_idxs = []
        self.perturb = None

    def set_probes(self, anchors, perturb):
        """Set the new probes based on the calculated anchors."""
        self.update_state(perturb)
        for vector in anchors.vectors:
            self.create_probes(vector)
        # Sanity check
        assert len(self.vectors) == ((len(anchors.vectors))*(self.nb_probes))

    def update_state(self, perturb):
        """Updates class state."""
        self.vectors = []  # Reset state
        self.perturb = perturb

    def create_probes(self, vector):
        """Create a clone of anchors, then implement the perturbation operation.
        """
        for _ in range(self.nb_probes):
            probe = vector.clone()
            self.perturb.apply(probe)
            self.vectors.append(probe)


#
