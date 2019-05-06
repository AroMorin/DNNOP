"""Class for analysis operations on the scores."""
from __future__ import division
from .step_size import Step_size
import torch
import math
import time

class Integrity(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.step_size = Step_size()
        self.top = torch.tensor(self.hp.initial_score, device='cuda')
        self.score = torch.tensor(self.hp.initial_score, device='cuda')
        self.prev_score = torch.tensor(self.hp.initial_score, device='cuda')
        self.value = self.hp.initial_integrity
        self.entropy = 0
        self.improvement = False

    def set_integrity(self, score):
        """Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.score = score
        self.step_size.set_step_size(self.value)
        if not self.improved():
            self.improvement = False
            self.reduce_integrity()
            self.step_size.decrease_bin()

        else:  # Improved
            self.improvement = True
            self.maintain_integrity()
            self.step_size.increase_bin()

        if self.improved_abs():
            self.top = self.score

    def improved_abs(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.hp.minimizing:
            return self.score < self.top
        else:
            return self.score > self.top

    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        self.set_entropy()
        if self.hp.minimizing:
            return self.entropy <= self.hp.min_entropy
        else:
            return self.entropy >= self.hp.min_entropy

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        """
        eps = self.top.ne(0)
        if eps:
            # Percentage change
            i = torch.sub(self.score, self.top)
            i = torch.div(i, self.top.abs())
            i = torch.mul(i, 100)
            self.entropy = i
        else:
            # Prevent division by zero
            i = torch.sub(self.score, self.top)
            i = torch.div(i, self.hp.epsilon)
            i = torch.mul(i, 100)
            self.entropy = i

    def reduce_integrity(self):
        # Reduce integrity, but not below the minimum allowed level
        step_size = self.step_size.value
        a = self.value-step_size  # Decrease integrity
        if a <= self.hp.min_integrity:
            self.value = self.hp.max_integrity  # Reset integrity
        else:
            self.value = max(0, a)  # Integrity never below zero

    def maintain_integrity(self):
        a = self.value+0.01
        b = self.hp.max_integrity
        self.value = min(a, b)

    def suspend_reality(self):
        self.real_value = self.value
        self.real_score = self.score

    def restore_reality(self):
        self.value = self.real_integrity
        self.score = self.real_score


#
