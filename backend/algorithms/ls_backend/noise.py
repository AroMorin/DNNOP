"""Class for applying perturbation."""

from __future__ import division
import torch
import math

from .search_radius import SR
from .direction import Direction

class Noise(object):
    def __init__(self, hp, params):
        self.hp = hp
        self.sr = SR(hp)
        self.direction = Direction(hp, params.numel())
        self.vec_length = params.numel()
        print("Number of trainable parameters: %d" %self.vec_length)
        self.precision = params.dtype
        self.noise_distribution = "uniform"  # Or "uniform"
        self.distribution = None
        self.magnitude = torch.empty(self.direction.num_selections,
                                dtype=self.precision,
                                device=params.device)
        self.vector = torch.empty_like(params)

    def update_state(self, improved):
        self.direction.update_state(improved)
        self.sr.update_state(improved)
        self.set_noise()
        #self.set_vector()

    def set_noise(self):
        # Cast to precision and CUDA, and edit shape
        self.magnitude.uniform_(self.sr.min_val, self.sr.max_val).squeeze()

    def set_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        self.vector.fill_(0.)
        self.vector[self.direction.value] = self.magnitude



#
