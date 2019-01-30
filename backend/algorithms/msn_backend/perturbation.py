"""Apply Perturbation"""

from __future__ import division
import numpy as np
import torch
from torch.distributions import uniform

class Perturbation:
    def __init__(self, hp):
        self.hp = hp
        self.integrity = self.hp.initial_integrity
        self.noise_distribution = "normal"  # Or "uniform"
        self.noise_type = "continuous"  # Or "discrete"
        self.vec_length = 0
        self.indices = []
        self.size = 0
        self.distribution = None

    def set_perturbation(self, vec, analyzer):
        self.vec_length = torch.numel(vec)
        self.indices = range(self.vec_length)
        self.size = int(analyzer.num_selections*self.vec_length)
        print("Number of selections: %d" %self.size)
        a = -analyzer.search_radius
        b = analyzer.search_radius
        self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))

    def apply(self, vec):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        choices = np.random.choice(self.indices, self.size, replace = False)
        choices = torch.tensor(choices).cuda().long()
        #noise = temp.normal_(mean=0, std=analyzer.search_radius)
        #vec = vec.put_(choices, noise, accumulate=True)
        noise = torch.zeros((self.vec_length)).cuda().half()
        dist = self.distribution.sample(torch.Size([self.size])).cuda().half().squeeze()
        noise[choices] = dist
        vec.add_(noise)
        #vec[choices] = noise
        #vec.index_add_(0, choices, noise)
        # I either explicitly return vector or this is sufficient







#
