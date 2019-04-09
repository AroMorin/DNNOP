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
        self.distribution = None
        self.precision = None
        self.choices = []
        self.vanilla_delta = 0
        self.unifrom_p = None
        self.p = None  # Index choice probability vector (P dist)
        self.p_counter = 0
        self.decr = 0.1  # decrease is 10% of probability value
        self.incr = 0.2  # increase is 20% of probability value
        self.device = torch.device('cuda')

    def init_perturbation(self, vec):
        """Initialize state and some variables."""
        self.precision = vec.dtype
        self.vec_length = torch.numel(vec)
        self.indices = torch.arange(self.vec_length, device=self.device)
        self.vanilla_delta = 1/(self.vec_length)
        self.incr *= self.vanilla_delta  # Factor in the vector size
        self.decr *= self.vanilla_delta  # Factor in the vector size
        self.uniform_p = torch.full(self.vec_length, self.vanilla_delta)
        self.p = self.uniform_p

    def update_state(self, analyzer):
        # Set noise size (scope)
        self.choices = []
        self.size = int(analyzer.num_selections*self.vec_length)
        print("Number of selections: %d" %self.size)
        self.set_noise_dist(analyzer.search_radius)
        self.p_counter = 0  # Resets counter for P-dist function
        self.update_p(analzer.scores)

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

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        np.random.seed()
        choices = np.random.choice(self.indices, self.size, replace=False, p=self.p)
        self.choices.append(choices.tolist())

    def update_p(self, scores, top):
        """Updates the probability distribution."""
        for i, choices in enumerate(self.choices):
            if self.improved(self.scores[i], top):
                self.increase_p(choices)
            else:
                self.decrease_p(choices)

    def improved(self, a, top):
        if self.hp.minimizing:
            return a < top
        else:
            return a > top

    def increase_p(self, choices):
        """This method decreases p at "choices" locations. No need to worry
        about negatives here. We decrease by the same amount we increased.
        """
        # Pull up "choices"
        dt = torch.full(self.size, self.incr)  # Delta tensor
        self.p[choices].add_(dt)
        # Push down everything else
        others = np.delete(np.arange(self.vec_length), choices)
        delta = (1-self.p.sum().item())/len(others)
        dt = torch.full(len(others), delta)  # Delta tensor
        self.p[others].sub_(delta)

    def decrease_p(self, choices):
        """This method decreases p at "choices" locations."""
        # Push down "choices"
        dt = torch.full(self.size, self.decr)
        self.p[choices].sub_(delta)
        self.p[self.p<0] = 0  # No negative probabilities

        # Pull up everything else
        others = np.delete(np.arange(self.vec_length), choices)
        d = (1-self.p.sum().item())/len(others)
        delta = torch.full(len(others), d)
        self.p[others].add_(delta)


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
        basis[self.choices] = noise
        return basis







#
