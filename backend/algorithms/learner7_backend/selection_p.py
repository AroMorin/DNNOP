"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch

class Selection_P(object):
    def __init__(self, hp, length):
        self.hp = hp
        self.incr = 0.84 # increase is 20% of probability value
        self.decr = 0.0  # decrease is 10% of probability value
        self.variance = 0
        self.max_var = 3
        self.p_vec = torch.full((length,), 0.5, device='cuda')
        self.uniform_vec = torch.full((length,), 0.5, device='cuda')
        self.p = torch.nn.functional.softmax(self.uniform_vec, dim=0)
        self.choices = []
        self.step = 0
        self.max_steps = 1000

    def update_state(self, improved, choices):
        self.choices = choices
        self.update_p(improved)
        self.check_step()
        self.step+=1

    def update_p(self, improved):
        """Updates the probability distribution."""
        if improved:
            self.increase_p()
        else:
            self.decrease_p()
        self.p = torch.nn.functional.softmax(self.p_vec, dim=0)  # Normalize
        self.variance = np.var(self.p_vec.cpu().numpy())
        self.check_var()
        self.check_p()

    def increase_p(self):
        """This method decreases p at "choices" locations."""
        # Pull up "choices"
        # Delta tensor
        dt = torch.full_like(self.p_vec[self.choices], self.incr, device='cuda')
        self.p_vec[self.choices] = torch.add(self.p_vec[self.choices], dt)

    def decrease_p(self):
        """This method decreases p at "choices" locations."""
        # Push down "choices"
        dt = torch.full_like(self.p_vec[self.choices], self.decr, device='cuda')
        self.p_vec[self.choices] = torch.sub(self.p_vec[self.choices], dt)

    def check_var(self):
        if self.variance>self.max_var:
            self.p_vec = self.uniform_vec.clone()
            self.p = torch.nn.functional.softmax(self.p_vec, dim=0)  # Normalize

    def check_p(self):
        if len((self.p == 0).nonzero())>0:
            nb_zeros = len((self.p == 0).nonzero())
            print("Error: %d Zero elements in self.p" %nb_zeros)
            exit()

    def check_step(self):
        if self.step>self.max_steps:
            self.p_vec = self.uniform_vec.clone()
            self.p = torch.nn.functional.softmax(self.p_vec, dim=0)  # Normalize
            self.step = 0


#
