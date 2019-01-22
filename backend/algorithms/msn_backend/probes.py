"""Base class for probes, apply blends"""

from .perturbation import Perturbation

class Probes:
    def __init__(self, hp):
        self.hp = hp
        self.probes = []
        self.peturb = Perturbation(hp)

    def set_probes(self, anchors):
        clones = self.create_clones(anchors)
        for clone in clones:
            self.probes.append(self.perturb.apply(clone))
        # Sanity check
        assert len(self.probes) == ((len(anchors))*(self.hp.nb_probes))

    def create_clones(self, anchors):
        clones = []
        for anchor in anchors:
            for _ in range(self.hp.nb_probes):
                clone = anchor.clone()
                clones.append(clone)
        return clones
