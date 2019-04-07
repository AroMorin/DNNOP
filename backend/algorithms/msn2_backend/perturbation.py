"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
from torch.distributions import uniform, normal

class Perturbation(object):
    def __init__(self, hp):
        self.hp = hp
        self.integrity = self.hp.initial_integrity
        self.noise_distribution = "normal"  # Or "uniform"
        self.noise_type = "continuous"  # Or "discrete"
        self.vec_length = 0
        self.indices = []
        self.size = 0
        self.distribution = None
        self.precision = None

    def set_perturbation(self, vec, analyzer):
        """Determines the shape and magnitude of the noise."""
        self.precision = vec.dtype
        self.vec_length = torch.numel(vec)
        self.indices = range(self.vec_length)
        self.size = int(analyzer.num_selections*self.vec_length)
        print("Number of selections: %d" %self.size)
        a = -analyzer.search_radius
        b = analyzer.search_radius
        if self.noise_distribution == "uniform":
            self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))
        elif self.noise_distribution == "normal":
            self.distribution = normal.Normal(torch.Tensor([0]), torch.Tensor([b]))

    def apply_uni(self, vec):
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
        noise = torch.zeros((self.vec_length), dtype=self.precision).cuda()
        dist = self.distribution.sample(torch.Size([self.size]), dtype=self.precision)
        dist.cuda().squeeze()
        noise[choices] = dist
        vec.add_(noise)
        #vec[choices] = noise
        #vec.index_add_(0, choices, noise)
        # I either explicitly return vector or this is sufficient

    def apply(self, vec):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        np.random.seed()
        choices = np.random.choice(self.indices, self.size, replace=False)
        choices = torch.tensor(choices).cuda().long()
        #noise = temp.normal_(mean=0, std=analyzer.search_radius)
        #vec = vec.put_(choices, noise, accumulate=True)
        noise = torch.zeros((self.vec_length), dtype=self.precision, device=torch.device('cuda'))
        if self.precision == torch.float:
            dist = self.distribution.sample(torch.Size([self.size])).float()
        elif self.precision == torch.half:
            dist = self.distribution.sample(torch.Size([self.size])).half()
        else:
            print("Unknown precision type")
            exit()
        dist = dist.cuda().squeeze()
        noise[choices] = dist
        vec.add_(noise)



#
