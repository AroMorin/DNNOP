"""Class for analysis operations on the scores."""
from __future__ import division
from .step_size import Step_size
import torch
import math
import time

class Integrity(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.step_size = Step_size(hyper_params)
        self.value = self.hp.initial_integrity

    def set_integrity(self, improved):
        """Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.step_size.update(improved, self.value)
        if not improved:
            self.reduce_integrity()
        else:
            self.maintain_integrity()
            #self.reset_integrity()

    def reduce_integrity(self):
        """Reduce integrity, but not below the minimum allowed level."""
        step_size = self.step_size.value
        a = self.value-step_size  # Decrease integrity
        if a <= 0.01:  # 0.01 is the minimum integrity
            self.value = 0.99  # Reset integrity (0.99 is maximum)
        else:
            self.value = max(0.01, a)  # Integrity never below zero

    def maintain_integrity(self):
        a = self.value+0.00  # 0.25 pushes the integrity back a bin
        self.value = min(a, 0.99)

    def reset_integrity(self):
        self.value = 0.99


#
