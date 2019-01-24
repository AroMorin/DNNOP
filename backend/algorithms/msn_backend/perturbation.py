"""Apply Perturbation"""

from __future__ import division
import numpy as np
import torch
import random

class Perturbation:
    def __init__(self, hp):
        self.hp = hp
        self.search_radius = 0
        self.num_selections = 0
        self.integrity = self.hp.initial_integrity
        self.noise_distribution = "normal"  # Or "uniform"
        self.noise_type = "continuous"  # Or "discrete"

    def apply(self, vec, analyzer):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        vec_length = vec.size()
        indices = random.sample(range(vec_length), self.analyzer.num_selections)
        temp = torch.new_empty((self.analyzer.num_selections))
        noise = temp.normal(mean=0, std=self.analyzer.search_radius)
        vec.put_(indices=indices,tensor=noise, accumulate=True)
        # I either explicitly return vector or this is sufficient






#
