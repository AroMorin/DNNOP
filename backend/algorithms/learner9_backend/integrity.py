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
        self.entropy = 0
        self.improvement = False

    def set_integrity(self, improved):
        """Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.step_size.set_step_size(self.value)
        if not improved:
            self.reduce_integrity()
            self.step_size.decrease_bin(self.value)

        else:
            self.maintain_integrity()
            self.step_size.increase_bin(self.value)

    def reduce_integrity(self):
        # Reduce integrity, but not below the minimum allowed level
        step_size = self.step_size.value
        a = self.value-step_size  # Decrease integrity
        if a <= self.hp.min_integrity:
            self.value = self.hp.max_integrity  # Reset integrity
        else:
            self.value = max(0, a)  # Integrity never below zero

    def maintain_integrity(self):
        a = self.value+0.25
        b = self.hp.max_integrity
        self.value = min(a, b)

    def suspend_reality(self):
        self.real_value = self.value

    def restore_reality(self):
        self.value = self.real_integrity


#
