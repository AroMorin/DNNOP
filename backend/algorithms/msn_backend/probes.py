"""Base class for probes, apply blends"""

from .perturbation import Perturbation

class Probes:
    def __init__(self, hp):
        self.nb_probes = hp.nb_probes
        self.models = []

    def set_probes(self, anchors, analyzer):
        self.models = []  # Reset state
        self.create_clones(anchors)
        # Sanity check
        assert len(self.models) == ((len(anchors))*(self.nb_probes))

    def create_clones(self, anchors):
        for anchor in anchors:
            for _ in range(self.nb_probes):
                clone = anchor.clone()
                self.models.append(clone)
