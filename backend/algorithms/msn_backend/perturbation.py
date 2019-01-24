"""Apply Perturbation"""

from __future__ import division
import numpy as np

class Perturbation:
    def __init__(self, hp):
        self.hp = hp
        self.search_radius = 0
        self.num_selections = 0
        self.integrity = self.hp.initial_integrity
        self.noise_type = "discrete"  # Or "continuous"

    def apply(self, vec, integrity):
        self.integrity = integrity
        self.set_num_selections()

    def set_num_selections(self):
        p = 1-self.integrity
        numerator = self.hp.alpha
        denominator = 1+(self.hp.beta/p)
        self.num_selections = numerator/denominator

    def set_search_radius(self):
        p = 1-self.integrity
        argument = (self.hp.lamda*p)-2.5
        exp1 = np.tanh(argument)+1
        self.search_radius = exp1*self.hp.lr

    def noise(self):
















#
