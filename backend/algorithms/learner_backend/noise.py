"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math
from torch.distributions import uniform, normal

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
        self.noise = None
        self.vector = None

    def update_state(self, integrity, improved):
        self.direction.update_state(integrity, improved)
        self.sr.update_state(integrity)
        #self.set_noise_dist()
        self.set_noise()
        self.set_vector()

    def set_noise_dist(self):
        """Determines the shape and magnitude of the noise."""
        a = self.sr_min
        b = self.sr_max
        c = (b-a)/2.
        assert a != b  # Sanity check
        if self.noise_distribution == "uniform":
            self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))
        elif self.noise_distribution == "normal":
            self.distribution = normal.Normal(torch.Tensor([c]), torch.Tensor([b]))
        else:
            print("Unknown distribution type")
            exit()

    def set_noise(self):
        #noise = self.distribution.sample(torch.Size([self.num_selections]))
        # Cast to precision and CUDA, and edit shape
        noise = torch.empty(self.direction.num_selections, dtype=self.precision,
                            device='cuda')
        self.noise = noise.uniform_(self.sr.min_val, self.sr.max_val).squeeze()
        #self.noise = noise.to(dtype=self.precision, device='cuda').squeeze()

    def set_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise_vector = torch.zeros(self.vec_length, dtype=self.precision,
                                    device='cuda')
        noise_vector[self.direction.value] = self.noise
        self.vector = noise_vector



#
