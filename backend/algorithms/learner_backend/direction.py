"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math

class Direction(object):
    def __init__(self, hp, vector_length):
        self.hp = hp
        self.vec_length = vector_length
        print("Number of trainable parameters: %d" %self.vec_length)
        self.indices = np.arange(self.vec_length)
        self.running_idxs = np.arange(self.vec_length)
        self.value = []  # list of indices
        self.limit = int(0.01*self.vec_length)
        self.num_selections = 500
        self.counter = 1
        self.step = 0

    def update_state(self, integrity, improved):
        if self.step==0:
            #self.set_num_selections(integrity)
            self.set_choices()
            self.counter = 1
        if improved:
            self.step += int(1/self.counter)
            self.counter+=1
        step = max(0, self.step-1)
        self.step = step
        print("Remaining steps: %d" %self.step)

    def set_num_selections(self, integrity):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        p = 1.-integrity
        #p = integrity
        argument = (5*p)-3.5
        exp1 = math.tanh(argument)+1
        self.num_selections = max(1, int(exp1*0.5*self.limit))

    def set_choices_(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idxs()
        choices = np.random.choice(self.running_idxs, self.num_selections)
        self.running_idxs = np.delete(self.running_idxs, choices)
        self.choices = choices.tolist()

    def check_idxs(self):
        if len(self.running_idxs)<self.num_selections:
            self.running_idxs = np.arange(self.vec_length)

    def set_choices(self):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        #p = p.cpu().numpy()  # Casting
        #self.choices = np.random.choice(self.indices, self.num_selections,
        #                                replace=False)
        choices = np.random.randint(0, self.vec_length, self.num_selections*2)
        choices = np.unique(choices)
        self.choices = choices[0:self.num_selections]



#
