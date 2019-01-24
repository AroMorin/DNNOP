"""Base class for probes, apply blends"""

from .perturbation import Perturbation

class Probes:
    def __init__(self, hp):
        self.hp = hp
        self.probes = []
        self.radial_expansion = False
        self.nb_anchors = 0

    def set_probes(self, anchors):
        self.nb_anchors = len(anchors)
        self.probes = []  # Reset state
        self.create_clones(anchors)
        self.set_radial_expansion()
        # Sanity check
        assert len(self.probes) == ((len(anchors))*(self.hp.nb_probes))

    def create_clones(self, anchors):
        for anchor in anchors:
            for _ in range(self.hp.nb_probes):
                clone = anchor.clone()
                self.probes.append(clone)

    def set_radial_expansion(self):
        """Triggers radial expansion only if the condition is met. Then in the
        new turn it switches it off again.
        It compares the actual number of anchors with the desired number of
        anchors
        """
        self.radial_expansion = False
        if self.nb_anchors < self.hp.nb_anchors:
            self.radial_expansion = True
            print("--Expanding Search Radius!--")
