"""Class for applying perturbation."""

from __future__ import division
import numpy as np

class Direction(object):
    def __init__(self, hp, vector_length):
        self.hp = hp
        self.vec_length = vector_length
        self.limit = 0.1
        self.idx = 0
        self.running_idxs = np.arange(self.vec_length)
        np.random.shuffle(self.running_idxs)  # To start sampling indices
        self.step = 0
        self.value = []  # list of indices
        self.counter = 1
        self.num_selections = 25
        self.b = 1  # bonus to steps
        self.f = 0.00005

    def update_state(self, improved):
        self.set_value()
        #if self.step==0:
            #self.set_num_selections(integrity)
            #self.set_value()
            #self.counter = 1
            #self.step = 10
        #self.update_step(improved)

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
        self.check_idx()
        choices = self.running_idxs[self.idx:self.idx+self.num_selections]
        self.value = choices
        self.idx+=self.num_selections

    def check_idx(self):
        if (self.idx+self.num_selections)>self.vec_length:
            self.idx=0
            np.random.shuffle(self.running_idxs)

    def update_step(self, improved):
        if improved:
            self.step += int(self.b/self.counter)
            self.counter+=1
        step = max(0, self.step-1)
        self.step = step
        print("Remaining steps: %d" %self.step)



#
