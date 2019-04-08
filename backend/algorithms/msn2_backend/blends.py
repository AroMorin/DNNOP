"""Class that defines all blend operations."""

from __future__ import division
#from random import choices  # Unavailable in python 2
import torch
import numpy as np

class Blends(object):
    def __init__(self, hp):
        self.hp = hp
        self.nb_blends = self.hp.pool_size-((self.hp.nb_anchors*self.hp.nb_probes)+1)
        self.nb_anchors = 0  # State not hyperparameter
        self.vectors = []
        self.blends_idxs = []
        self.blend_type = "crisscross"  # Or "random choice"
        self.anchors = None
        self.analyzer = None
        self.vec_length = 0
        self.compounds1 = []
        self.compounds2 = []
        self.indices = []

    def set_blends(self, anchors, vectors, analyzer, perturb):
        """Sets the blends number and vectors."""
        self.update_state(anchors, analyzer, perturb)
        if self.nb_blends>0:
            self.set_indices()
            self.set_compounds1()
            self.set_compounds2(vectors)
            self.blend()

    def update_state(self, anchors, analyzer, perturb):
        """Resets the class state and determines the number of blends."""
        self.vectors = [] # Reset state
        self.anchors = anchors
        self.analyzer = analyzer
        self.perturb = perturb
        self.vec_length = torch.numel(anchors.vectors[0])
        self.nb_anchors = len(anchors.vectors)
        self.nb_blends = self.hp.pool_size-(self.nb_anchors+(
                                    self.nb_anchors*self.hp.nb_probes)+1)

    def set_indices(self):
        """Sets the indices for the blends. It has two options, either a
        checkered pattern, or a random pattern (for blending).
        """
        # In case I wanted a variable blending method
        #indices = random.sample(range(self.vec_length), self.analyzer.num_selections)
        #self.indices = random.sample(range(self.vec_length), self.vec_length/2)
        # I can select/determine a random sequence, and keep it for the iteration
        self.indices = np.arange(start=0, stop=self.vec_length, step=2)
        self.indices = torch.tensor(self.indices).cuda().long()

    def set_compounds1(self):
        """Random choices from anchors, with replacement, as the lineup of first
        blend components.
        """
        # From anchors
        #idxs = choices(range(self.nb_anchors), k=self.nb_blends)
        idxs = np.random.choice(range(self.nb_anchors), size=self.nb_blends, replace=True)
        self.compounds1 = [self.anchors.vectors[i] for i in idxs]

    def set_compounds2(self, vectors):
        """Random choices from probes+blends, with replacement, as the lineup
        of second blend components.
        """
        # From pool
        lower = self.anchors.nb_anchors+1
        upper = self.anchors.nb_anchors*self.hp.nb_probes
        #idxs = choices(range(lower, upper+1), k=self.nb_blends)
        idxs = np.random.choice(range(lower, upper+1), size=self.nb_blends, replace=True)
        self.compounds2 = [vectors[i] for i in idxs]

    def blend(self):
        """Implements the blend action between the first and second lineup of
        components.
        """
        for i in range(self.nb_blends):
            c1 = self.compounds1[i].clone()
            c2 = self.compounds2[i]
            c1.put_(self.indices, c2[self.indices])
            blend = c1
            self.perturb.apply(blend)
            self.vectors.append(blend)
