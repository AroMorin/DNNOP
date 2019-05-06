"""Class for analysis operations on the scores."""

from __future__ import division

class Step_size(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.value = self.hp.step_size
        self.bin = [1, 1, 1, 1]  # Uniform distribution
        self.real_bin = [1, 1, 1, 1]
        self.max_steps = 20  # Max number of iterations to spend in one bin

    def set_step_size(self, integrity):
        if  0. < integrity <= 0.25:
            a = 0.25/self.bin[0]
            self.value = min(a, self.max_steps)
        elif  0.25 < integrity <= 0.5:
            a = 0.25/self.bin[1]
            self.value = min(a, self.max_steps)
        elif  0.5 < integrity <= 0.75:
            a = 0.25/self.bin[2]
            self.value = min(a, self.max_steps)
        elif  0.75 < integrity <= 1.:
            a = 0.25/self.bin[3]
            self.value = min(a, self.max_steps)

    def increase_bin(self):
        if  0. < self.value <= 0.25:
            self.bin[0]+= 2
        elif  0.25 < self.value <= 0.5:
            self.bin[1]+= 2
        elif  0.5 < self.value <= 0.75:
            self.bin[2]+= 2
        elif  0.75 < self.value <= 1.:
            self.bin[3]+= 2

    def decrease_bin(self):
        if  0. < self.value <= 0.25:
            self.bin[0] = max(1, self.bin[0]-1)
        elif  0.25 < self.value <= 0.5:
            self.bin[1] = max(1, self.bin[1]-1)
        elif  0.5 < self.value <= 0.75:
            self.bin[2] = max(1, self.bin[2]-1)
        elif  0.75 < self.value <= 1.:
            self.bin[3] = max(1, self.bin[3]-1)

    def suspend_reality(self):
        self.real_bin = self.bin

    def restore_reality(self):
        self.bin = self.real_bin
