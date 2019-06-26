"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math
from torch.distributions import uniform, normal

class SR(object):
    def __init__(self, hp):
        self.hp = hp
        self.noise_distribution = "uniform"  # Or "uniform"
        self.distribution = None
        self.gamma = 0.000012
        #self.decay = 0.000005
        self.min_val = -0.1
        self.max_val = 0.1
        self.lr = 0.3
        self.inc = 0.01
        self.dec = 0.01

    def update_state(self, integrity, improved):
        if improved:
            self.increment()
        else:
            self.decrement()
        self.decay()
        self.set_value(integrity)

    def increment(self):
        self.lr += self.inc

    def decrement(self):
        self.lr -= self.dec

    def decay(self):
        lr = self.lr-self.gamma
        self.lr = max(0.05, lr)  # min learning rate

    def set_value(self, integrity):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = integrity
        #p = 1.-integrity
        argument = (5*p)-2.0
        exp1 = math.tanh(argument)+1
        self.min_val = -exp1*self.lr
        self.max_val = exp1*self.lr
        print("LR: %f " %self.lr)



#
