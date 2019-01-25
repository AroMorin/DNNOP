"""Class that defines all blend operations."""

from __future__ import division
import random
import torch

class Blends:
    def __init__(self, hp):
        self.nb_blends = hp.pool_size-((hp.nb_anchors*hp.nb_probes)+1)
        self.models = []
        self.nb_anchors = 0
        self.blend_type = "crisscross"  # Or "random choice"
        self.indices = []

    def set_blends(self, anchors, current_pool, analyzer):
        self.models = [] # Reset state
        self.anchors = anchors
        self.current_pool = current_pool
        self.analyzer = analyzer
        self.vec_length = anchors[0].size()
        self.pool_size = len(current_pool)
        self.nb_anchors = len(anchors)
        self.nb_blends = self.pool_size-(self.nb_anchors*self.hp.nb_probes)
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

    def select_anchors(self):
        self.parents1 = random.sample(range(self.nb_anchors), self.nb_blends)

    def select_parents2(self):
        self.parents2 = random.sample(range(self.pool_size), self.nb_blends)

    def blend(self):
        for i in range(self.nb_blends):
            p1 = self.anchors[self.parents1[i]]
            p2 = self.current_pool[self.parents2[i]]
            p1.put_(indices=self.indices, tensor=p2)  # Accumulate false
            self.models.append(p1)
