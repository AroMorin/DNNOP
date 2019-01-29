"""Base class for probes, apply blends"""

from .perturbation import Perturbation

class Probes:
    def __init__(self, hp):
        self.nb_probes = hp.nb_probes
        self.models = []
        self.perturb = None
        self.anchors = None

    def set_probes(self, anchors, perturb):
        self.update_state(anchors, perturb)
        for anchor in anchors.models:
            self.create_probes(anchor)
        # Sanity check
        assert len(self.models) == ((len(anchors.models))*(self.nb_probes))

    def update_state(self, anchors, perturb):
        self.models = []  # Reset state
        self.anchors = anchors
        self.perturb = perturb

    def create_probes(self, anchor):
        for _ in range(self.nb_probes):
            probe = anchor.clone()
            self.perturb.apply(probe)
            self.models.append(probe)






#
