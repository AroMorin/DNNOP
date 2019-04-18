"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
from torch.distributions import uniform, normal

class Perturbation(object):
    def __init__(self, hp):
        self.hp = hp
        self.noise_distribution = "uniform"  # Or "uniform"
        self.noise_type = "continuous"  # Or "discrete" -- Unimplemented feature
        self.vec_length = 0
        self.indices = []
        self.size = 0
        self.score = self.hp.initial_score
        self.prev_score = self.hp.initial_score
        self.distribution = None
        self.precision = None
        self.choices = []  # list of indices
        self.vd = 0  # p value for uniform distribution
        self.unifrom_p = None
        self.device = torch.device('cuda')
        self.p_vec = None
        self.p = None  # Index choice probability vector (P dist)
        self.p_counter = 0
        self.decr = 0.01  # decrease is 10% of probability value
        self.incr = 0.1  # increase is 20% of probability value
        self.variance = 0

    def init_perturbation(self, vec):
        """Initialize state and some variables."""
        self.precision = vec.dtype
        self.vec_length = torch.numel(vec)
        self.indices = np.arange(self.vec_length)
        self.p_vec = torch.full((self.vec_length,), 0.5, device=self.device)
        self.uniform_vec = torch.full((self.vec_length,), 0.5, device=self.device)
        self.p = torch.nn.functional.softmax(self.uniform_vec, dim=0)

    def update_state(self, analyzer):
        # Set noise size (scope)
        self.choices = []
        self.size = int(analyzer.num_selections*self.vec_length)
        self.set_noise_dist(analyzer.search_radius)
        self.set_choices()
        if 0 < self.p_counter:  # Reset p every 100 iterations
            self.score = analyzer.score  # Acquire new state
            self.update_p()
            self.prev_score = self.score  # Update state
            self.p_counter+=1
        elif self.p_counter == 0:
            self.p_counter+=1  # Need to step at least once

    def set_noise_dist(self, limit):
        """Determines the shape and magnitude of the noise."""
        a = -limit
        b = limit
        if self.noise_distribution == "uniform":
            self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))
        elif self.noise_distribution == "normal":
            self.distribution = normal.Normal(torch.Tensor([0]), torch.Tensor([b]))
        else:
            print("Unknown precision type")
            exit()

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        np.random.seed()
        if len((self.p == 0).nonzero())>0:
            nb_zeros = len((self.p == 0).nonzero())
            print("Error: %d Zero elements in self.p" %nb_zeros)
            exit()
        self.choices = np.random.choice(self.indices, self.size, replace=False, p=self.p.cpu().numpy())

    def update_p(self):
        """Updates the probability distribution."""
        if self.improved():
            self.increase_p()
        else:
            self.decrease_p()
        self.p = torch.nn.functional.softmax(self.p_vec, dim=0)  # Normalize
        self.variance = np.var(self.p_vec.cpu().numpy())
        self.check_var()

    def improved(self):
        if self.hp.minimizing:
            return self.score < self.prev_score
        else:
            return self.score > self.prev_score

    def increase_p(self):
        """This method decreases p at "choices" locations."""
        # Pull up "choices"
        dt = torch.full((self.size,), self.incr, device=self.device)  # Delta tensor
        self.p_vec[self.choices] = torch.add(self.p_vec[self.choices], dt)

    def decrease_p(self):
        """This method decreases p at "choices" locations."""
        # Push down "choices"
        dt = torch.full((self.size,), self.decr, device=self.device)
        self.p_vec[self.choices] = torch.sub(self.p_vec[self.choices], dt)

    def check_var(self):
        if self.variance>2:
            self.p_vec = self.uniform_vec.clone()
            self.p = torch.nn.functional.softmax(self.p_vec, dim=0)  # Normalize

    def apply(self, vec):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        noise = self.get_noise()
        vec.add_(noise)

    def get_noise(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise = self.distribution.sample(torch.Size([self.size]))
        # Cast to precision and CUDA, and edit shape
        noise = noise.to(dtype=self.precision, device=self.device).squeeze()
        noise_vector = torch.zeros((self.vec_length), dtype=self.precision, device=self.device)
        noise_vector[self.choices] = noise
        return noise_vector

    def suspend_reality(self):
        self.real_p = self.p
        self.real_p_vec = self.p_vec
        self.real_variance = self.variance

    def restore_reality(self):
        self.p = self.real_p
        self.p_vec = self.real_p_vec
        self.variance = self.real_variance



#
