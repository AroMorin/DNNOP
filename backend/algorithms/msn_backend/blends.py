"""Class that defines all blend operations."""

from __future__ import division
from random import choices
import torch
import numpy as np

class Blends:
    def __init__(self, hp):
        self.hp = hp
        self.nb_blends = self.hp.pool_size-((self.hp.nb_anchors*self.hp.nb_probes)+1)
        self.nb_anchors = 0  # State not hyperparameter
        self.models = []
        self.blend_type = "crisscross"  # Or "random choice"
        self.anchors = []
        self.pool = []
        self.analyzer = None
        self.vec_length = 0
        self.idxs1 = []
        self.idxs2 = []
        self.indices = []

    def set_blends(self, anchors, pool, analyzer):
        self.models = [] # Reset state
        self.anchors = anchors
        self.pool = pool
        self.analyzer = analyzer
        self.vec_length = torch.numel(anchors[0])
        self.nb_anchors = len(anchors)
        self.nb_blends = self.hp.pool_size-(self.nb_anchors+(
                                    self.nb_anchors * self.hp.nb_probes))
        if self.nb_blends>0:
            self.select_indices()
            self.select_parents1()
            self.select_parents2()
            self.blend()

    def select_indices(self):
        # In case I wanted a variable blending method
        #indices = random.sample(range(self.vec_length), self.analyzer.num_selections)
        #self.indices = random.sample(range(self.vec_length), self.vec_length/2)
        # I can select/determine a random sequence, and keep it for the iteration
        self.indices = np.arange(start=0, stop=self.vec_length, step=2)
        self.indices = torch.tensor(self.indices).cuda().long()

    def select_parents1(self):
        # From anchors
        self.idxs1 = choices(range(self.nb_anchors), k=self.nb_blends)

    def select_parents2(self):
        # From pool
        self.idxs2 = choices(range(self.hp.pool_size), k=self.nb_blends)

    def blend(self):
        for i in range(self.nb_blends):
            p1 = self.anchors[self.idxs1[i]]
            p2 = self.pool[self.idxs2[i]]
            p2 = torch.take(p2, self.indices)
            p1.put_(self.indices, p2)  # Accumulate false
            self.models.append(p1)
