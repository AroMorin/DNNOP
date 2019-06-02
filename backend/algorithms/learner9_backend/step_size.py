"""Class for analysis operations on the scores."""

from __future__ import division

class Step_size(object):
    def __init__(self, hp):
        self.value = 0
        self.increment = 2
        self.decrement = 1
        self.min_steps = 1
        self.bin = [self.min_steps]*4  # Uniform distribution
        self.max_steps = hp.max_steps  # Max number of iterations to spend in one bin

    def update(self, improved, integrity):
        self.set_interval(integrity)
        self.set_step_size()
        if not improved:
            if integrity == 0.99:
                self.increase_depth()
            self.decrease_bin()
        else:
            self.increase_bin()
            self.reset_depth()

    def set_interval(self, integrity):
        if  0. < integrity <= 0.25:
            self.interval = 0
        elif  0.25 < integrity <= 0.5:
            self.interval = 1
        elif  0.5 < integrity <= 0.75:
            self.interval = 2
        elif  0.75 < integrity <= 1.:
            self.interval = 3

    def set_step_size(self):
        self.value = 0.25/self.bin[self.interval]

    def decrease_bin(self):
        a = self.bin[self.interval]-self.decrement
        self.bin[self.interval] = max(self.min_steps, a)

    def increase_depth(self):
        self.min_steps *= 2

    def increase_bin(self):
        a = self.bin[self.interval]+self.increment
        self.bin[self.interval] = min(a, self.max_steps)

    def reset_depth(self):
        self.min_steps = 1
