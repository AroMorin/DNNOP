"""Base class for probes"""

from .perturbation import Perturbation
import torch

class Probes(object):
    def __init__(self, hp):
        self.nb_probes = hp.nb_probes
        self.models = []
        self.probes_idxs = []
        self.perturb = None
        self.anchors = None

    def set_probes(self, anchors, perturb):
        """Set the new probes based on the calculated anchors."""
        self.update_state(anchors, perturb)
        for anchor in anchors.models:
            self.create_probes(anchor)
        # Sanity check
        assert len(self.models) == ((len(anchors.models))*(self.nb_probes))

    def update_state(self, anchors, perturb):
        """Updates class state."""
        self.models = []  # Reset state
        self.anchors = anchors
        self.perturb = perturb

    def create_probes(self, anchor):
        """Create a clone of anchors, then implement the perturbation operation.
        """
        for _ in range(self.nb_probes):
            probe = anchor.clone()
            self.perturb.apply(probe)
            self.models.append(probe)


#
