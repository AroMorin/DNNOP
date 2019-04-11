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
        self.noise_type = "continuous"  # Or "discrete" -- Unimplemented feature
        self.vec_length = 0
        self.indices = []
        self.size = 0
        self.scores = [self.hp.initial_score]*self.hp.pool_size
        self.prev_scores = [self.hp.initial_score]*self.hp.pool_size
        self.distribution = None
        self.precision = None
        self.choices = []
        self.vd = 0  # p value for uniform distribution
        self.unifrom_p = None
        self.p = None  # Index choice probability vector (P dist)
        self.p_counter = 0
        self.decr = 0.0  # decrease is 10% of probability value
        self.incr = 0.1  # increase is 20% of probability value
        self.device = torch.device('cuda')
        self.idx = 0

    def init_perturbation(self, vec):
        """Initialize state and some variables."""
        self.precision = vec.dtype
        self.vec_length = torch.numel(vec)
        self.indices = np.arange(self.vec_length)
        self.incr *= 0.5
        self.decr *= 0.5
        self.uniform_p = torch.nn.functional.softmax(
                        torch.full((self.vec_length,), 0.5, device=self.device),
                        dim=0)
        self.p = self.uniform_p

    def update_state(self, analyzer):
        # Set noise size (scope)
        self.choices = []
        self.size = int(analyzer.num_selections*self.vec_length)
        print("Number of selections: %d" %self.size)
        self.set_noise_dist(analyzer.search_radius)
        if self.p_counter >0 and self.p_counter < 500:  # Reset p every 100 iterations
            self.scores = analyzer.scores  # Acquire new state
            self.update_p()
            self.prev_scores = self.scores  # Update state
            self.p_counter += 1
        else:
            if self.p_counter == 0:
                self.p_counter += 1
            else:
                self.p = self.uniform_p
        self.idx = 0  # Reset state

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

    def apply(self, vec):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        self.set_choices()
        noise = self.get_noise()
        vec.add_(noise)
        self.idx+=1

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        np.random.seed()
        choices = np.random.choice(self.indices, self.size, replace=False, p=self.p.cpu().numpy())
        self.choices.append(choices.tolist())

    def update_p(self):
        """Updates the probability distribution."""
        for i, choices in enumerate(self.choices):
            print(i)
            if self.improved(i):
                self.increase_p(choices)
            else:
                self.decrease_p(choices)
        self.p = torch.nn.functional.softmax(self.p, dim=0)  # Normalize

    def improved(self, idx):
        if self.hp.minimizing:
            return self.scores[idx] < self.prev_scores[idx]
        else:
            return self.scores[idx] > self.prev_scores[idx]

    def increase_p(self, choices):
        """This method decreases p at "choices" locations."""
        # Pull up "choices"
        dt = torch.full((self.size,), self.incr, device=self.device)  # Delta tensor
        self.p[choices].add_(dt)

    def decrease_p(self, choices):
        """This method decreases p at "choices" locations."""
        # Push down "choices"
        dt = torch.full((self.size,), self.decr, device=self.device)
        self.p[choices].sub_(delta)

    def get_noise(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise = self.distribution.sample(torch.Size([self.size]))
        # Cast to precision and CUDA, and edit shape
        noise = noise.to(dtype=self.precision, device=self.device).squeeze()
        basis = torch.zeros((self.vec_length), dtype=self.precision, device=self.device)
        basis[self.choices[self.idx]] = noise
        return basis







#
