"""Class for analysis operations on the scores."""

from __future__ import division

class Step_size(object):
    def __init__(self, hp):
        self.value = 0.
        self.increment = 10
        self.decrement = 1
        self.min_steps = 3.
        self.bin = [self.min_steps]*4  # Uniform distribution
        self.real_bin = [self.min_steps]*4
        self.max_steps = hp.max_steps  # Max number of iterations to spend in one bin

    def set_step_size(self, integrity):
        if  0. < integrity <= 0.25:
            self.value = 0.25/self.bin[0]
        elif  0.25 < integrity <= 0.5:
            self.value = 0.25/self.bin[1]
        elif  0.5 < integrity <= 0.75:
            self.value = 0.25/self.bin[2]
        elif  0.75 < integrity <= 1.:
            self.value = 0.25/self.bin[3]

    def increase_bin(self, integrity):
        if  0. < integrity <= 0.25:
            a = self.bin[0]+self.increment
            self.bin[0] = min(a, self.max_steps)
        elif  0.25 < integrity <= 0.5:
            a = self.bin[1]+self.increment
            self.bin[1] = min(a, self.max_steps)
        elif  0.5 < integrity <= 0.75:
            a = self.bin[2]+self.increment
            self.bin[2] = min(a, self.max_steps)
        elif  0.75 < integrity <= 1.:
            a = self.bin[3]+self.increment
            self.bin[3] = min(a, self.max_steps)

    def decrease_bin(self, integrity):
        if  0. < integrity <= 0.25:
            self.bin[0] = max(self.min_steps, self.bin[0]-self.decrement)
        elif  0.25 < integrity <= 0.5:
            self.bin[1] = max(self.min_steps, self.bin[1]-self.decrement)
        elif  0.5 < integrity <= 0.75:
            self.bin[2] = max(self.min_steps, self.bin[2]-self.decrement)
        elif  0.75 < integrity <= 1.:
            self.bin[3] = max(self.min_steps, self.bin[3]-self.decrement)

    def suspend_reality(self):
        self.real_bin = self.bin

    def restore_reality(self):
        self.bin = self.real_bin
