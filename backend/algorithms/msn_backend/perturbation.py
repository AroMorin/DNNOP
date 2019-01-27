"""Apply Perturbation"""

from __future__ import division
import numpy as np
import torch

class Perturbation:
    def __init__(self, hp):
        self.hp = hp
        self.search_radius = 0
        self.num_selections = 0
        self.integrity = self.hp.initial_integrity
        self.noise_distribution = "normal"  # Or "uniform"
        self.noise_type = "continuous"  # Or "discrete"
        self.vec_length = 0

    def set_perturbation(self):
        pass

    def apply(self, vec, analyzer):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        self.vec_length = torch.numel(vec)
        indices = range(self.vec_length)
        size = int(analyzer.num_selections)
        choices = np.random.choice(indices, size, replace = False)
        choices = torch.tensor(choices).cuda().long()
        temp = torch.empty((size))
        noise = temp.normal_(mean=0, std=analyzer.search_radius).cuda().half()
        vec.put_(choices, noise, accumulate=True)
        # I either explicitly return vector or this is sufficient






#
