"""Class for applying perturbation."""

from __future__ import division
import torch
import math

class SR(object):
    def __init__(self, hp):
        self.hp = hp
        self.gamma = 0.000001
        #self.gamma = 0.000001
        self.lr = 0.5
        self.min_val = -self.lr
        self.max_val = self.lr
        self.inc = 0.00004
        self.dec = 0.00002
        self.min_limit = 0.07
        self.max_limit = 0.5

    def update_state(self, improved):
        #self.decay()
        self.min_val = -self.lr
        self.max_val = self.lr

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



#
