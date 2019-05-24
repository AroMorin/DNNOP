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
        self.noise_distribution = "uniform"  # Or "uniform"
        self.distribution = None
        self.choices = []  # list of indices
        self.limit = 5000
        self.num_selections = None
        self.search_radius = None
        self.precision = vector.dtype
        self.vector = None

    def update_state(self, integrity):
        # Set noise size (scope)
        self.choices = []
        self.set_num_selections(integrity)
        self.set_search_radius(integrity)
        self.set_noise_dist()
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

    def set_search_radius(self, integrity):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = 1.-integrity
        argument = (5.*p)-2.5
        exp1 = math.tanh(argument)+1
        self.search_radius = exp1*0.02

    def set_noise_dist(self):
        """Determines the shape and magnitude of the noise."""
        a = -self.search_radius
        b = self.search_radius
        if self.noise_distribution == "uniform":
            self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))
        elif self.noise_distribution == "normal":
            self.distribution = normal.Normal(torch.Tensor([0.]), torch.Tensor([b]))
        else:
            print("Unknown distribution type")
            exit()

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idxs()
        np.random.seed()
        self.choices = np.random.choice(self.running_idxs, self.num_selections)
        self.running_idxs = np.delete(self.running_idxs, self.choices)
        self.choices = torch.tensor(self.choices)

    def check_idxs(self):
        if len(self.running_idxs)<self.num_selections:
            self.running_idxs = np.arange(self.vec_length)

    def set_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise = self.distribution.sample(torch.Size([self.num_selections]))
        # Cast to precision and CUDA, and edit shape
        noise = noise.squeeze()
        noise_vector = torch.zeros(self.vec_length)
        noise_vector[self.choices] = noise
        noise_vector = noise_vector.to(dtype=self.precision, device='cuda')
        self.vector = noise_vector



#
