"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math
from torch.distributions import uniform, normal

class Noise(object):
    def __init__(self, hp, vector):
        self.hp = hp
        self.vec_length = torch.numel(vector)
        self.indices = np.arange(self.vec_length)
        self.running_idxs = np.arange(self.vec_length)
        self.choices = []  # list of indices
        self.limit = 10000
        self.num_selections = None
        self.precision = vector.dtype
        self.vector = None

    def update_state(self, integrity):
        # Set noise size (scope)
        self.choices = []
        self.set_num_selections(integrity)
        self.set_choices()
        self.set_vector()

    def set_num_selections(self, integrity):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        #p = 1-self.integrity
        p = integrity
        numerator = 1
        denominator = 1+(0.29/p)
        num_selections = numerator/denominator
        self.num_selections = int(num_selections*self.limit)

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idxs()
        np.random.seed()
        self.choices = np.random.choice(self.running_idxs, self.num_selections)
        self.running_idxs = np.delete(self.running_idxs, self.choices)
        self.choices = torch.tensor(self.choices).cuda()

    def check_idxs(self):
        if len(self.running_idxs)<self.num_selections:
            self.running_idxs = np.arange(self.vec_length)

    def set_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise = np.random.choice([0., 1.], size=self.num_selections)
        noise = torch.tensor(noise)
        # Cast to precision and CUDA, and edit shape
        self.vector = noise.to(dtype=self.precision, device='cuda').squeeze()
        #noise = torch.full(self.num_selections, 0.05, dtype=self.precision,
        #                            device='cuda')
        #noise_vector = torch.zeros(self.vec_length, dtype=self.precision,
        #                            device='cuda')
        #noise_vector[self.choices] = noise
        #self.vector = noise_vector



#
