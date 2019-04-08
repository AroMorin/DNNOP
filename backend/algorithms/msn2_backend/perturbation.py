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
        self.choices = None
        self.even_p = None
        self.p = None  # Index choice probability vector (P dist)
        self.p_counter = 0
        self.discount = 0.1  # discount is 10% of equal probability value
        self.device = torch.device('cuda')

    def init_perturbation(self, vec):
        """Initialize state and some variables."""
        self.precision = vec.dtype
        self.vec_length = torch.numel(vec)
        self.indices = range(self.vec_length)
        even_prob = 1/(self.vec_length)
        self.even_p = np.full(self.vec_length, even_prob)
        self.p = self.even_p
        self.discount = even_prob*self.discount  # Factor in the vector size

    def update_state(self, analyzer):
        # Set noise size (scope)
        self.size = int(analyzer.num_selections*self.vec_length)
        print("Number of selections: %d" %self.size)
        self.set_noise_dist(analyzer.search_radius)
        self.p_counter = 0  # Resets counter for P-dist function

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
        self.update_p(choices)
        #self.choices = torch.tensor(choices).cuda().long()
        self.choices = choices.tolist()

    def update_p(self, choices):
        """Counts the number of steps the function is called. We want to reset
        the P-distribution for every anchor. Thus, we reset the counter after
        M calls, which corresponds to M probes. To ensure this mechanism works
        as expected, we also reset counter every optimization iteration, to make
        sure we do not "carryover" the state into a new generation (since the
        number of anchors varies). We don't care how this behaves with blends.

        The "discount" constant defines the amount of depression.

        The function "depresses" the probability of selection distribution by
        creating a "depression vector" and subtracts said vector from the
        current p_vector. Subtraction happens only at the indices chosen.
        """
        if self.p_counter <= self.hp.nb_probes:
            self.push_down(choices)
            self.push_up(choices)
            self.p_counter += 1  # Increment counter
        else:
            self.p = self.even_p  # Reset State
            self.p_counter = 0  # Reset state, new anchor

    def push_down(self, choices):
        decrease = np.full(self.size, self.discount)
        temp = np.subtract(self.p[choices], decrease)
        temp[temp<0] = 0  # No negative probabilities
        self.p[choices] = temp

    def push_up(self, choices):
        others = np.delete(np.arange(self.vec_length), choices)
        increase = (1 - np.sum(self.p))/len(others)
        self.p[others] = np.add(self.p[others], increase)

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
