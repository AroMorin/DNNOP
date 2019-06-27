"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math

class Direction(object):
    def __init__(self, hp, vector_length):
        self.hp = hp
        self.vec_length = vector_length
        self.limit = int(0.01*self.vec_length)
        self.indices = np.arange(self.vec_length)
        self.running_idxs = np.arange(self.vec_length)
        self.step = 0
        self.value = []  # list of indices
        self.counter = 1
        self.num_selections = 50
        self.b = 1  # bonus to steps

    def update_state(self, integrity, improved):
        if self.step==0:
            #self.set_num_selections(integrity)
            self.set_value()
            self.counter = 1
        self.update_step(improved)

    def set_num_selections(self, integrity):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        p = 1.-integrity
        #p = integrity
        argument = (5*p)-3.5
        exp1 = math.tanh(argument)+1
        self.num_selections = max(1, int(exp1*0.5*self.limit))

    def set_value_(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        choices = np.random.randint(0, self.vec_length, self.num_selections*2)
        choices = np.unique(choices)
        self.value = choices[0:self.num_selections]

    def set_value(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idxs()
        choices = np.random.choice(self.running_idxs, self.num_selections)
        self.running_idxs = np.delete(self.running_idxs, choices)
        self.value = choices.tolist()

    def check_idxs(self):
        if len(self.running_idxs)<self.num_selections:
            self.running_idxs = np.arange(self.vec_length)

    def update_step(self, improved):
        if improved:
            self.step += int(self.b/self.counter)
            self.counter+=1
        step = max(0, self.step-1)
        self.step = step
        print("Remaining steps: %d" %self.step)



#
