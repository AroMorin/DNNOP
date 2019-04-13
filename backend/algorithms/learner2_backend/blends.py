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
        self.vector = None
        self.blends_idxs = []
        self.blend_type = "crisscross"  # Or "random choice"
        self.anchors = None
        self.analyzer = None
        self.vec_length = 0
        self.compound1 = None
        self.compound2 = None
        self.indices = []

    def update_state(self, anchors, analyzer, perturb):
        """Resets the class state and determines the number of blends."""
        self.vector = None # Reset state
        self.anchors = anchors
        self.analyzer = analyzer
        self.perturb = perturb
        self.vec_length = self.perturb.vec_length
        self.nb_anchors = anchors.nb_anchors
        self.nb_blends = self.hp.pool_size-(self.nb_anchors*self.hp.nb_probes)

    def generate(self, vector):
        """Sets the blends number and vectors."""
        assert self.nb_blends>0
        self.set_indices()
        self.set_compound1()
        self.blend(vector)

    def set_indices(self):
        """Sets the indices for the blends. It has two options, either a
        checkered pattern, or a random pattern (for blending).
        """
        # In case I wanted a variable blending method
        #indices = random.sample(range(self.vec_length), self.analyzer.num_selections)
        #self.indices = random.sample(range(self.vec_length), self.vec_length/2)
        # I can select/determine a random sequence, and keep it for the iteration
        self.indices = torch.arange(start=0, end=self.vec_length, step=2, device='cuda')

    def set_compound1(self):
        """Random choices from anchors, with replacement, as the lineup of first
        blend components.
        """
        # From anchors
        #idxs = choices(range(self.nb_anchors), k=self.nb_blends)
        idx = torch.randint(low=0, high=self.nb_anchors, size=(1,))
        self.compound1 = self.anchors.vectors[idx].clone()

    def blend(self, vector):
        """Implements the blend action between the first and second lineup of
        components.
        """
        c1 = self.compound1
        c1.put_(self.indices, vector[self.indices])
        self.perturb.apply(c1)
        self.perturb.apply(c1)  # Apply noise twice
        self.vector = c1







#