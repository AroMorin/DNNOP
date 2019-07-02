"""Class for applying perturbation."""

from __future__ import division
import torch
import math

class SR(object):
    def __init__(self, hp):
        self.hp = hp
        self.noise_distribution = "uniform"  # Or "uniform"
        self.distribution = None
        self.gamma = 0.00001
        #self.gamma = 0.000001
        self.min_val = -0.
        self.max_val = 0.
        self.lr = 0.4
        self.inc = 0.00004
        self.dec = 0.00002
        self.min_limit = 0.07
        self.max_limit = 0.5

    def update_state(self, integrity, improved):
        #self.adapt_lr(improved)
        self.decay()
        self.set_value(integrity)

    def adapt_lr(self, improved):
        if improved:
            self.increment()
        else:
            self.decrement()

    def increment(self):
        lr = self.lr+self.inc
        self.lr = min(self.max_limit, lr)  # min learning rate

    def decrement(self):
        lr = self.lr-self.dec
        self.lr = max(self.min_limit, lr)  # min learning rate

    def decay(self):
        lr = self.lr-self.gamma
        self.lr = max(self.min_limit, lr)  # min learning rate

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
