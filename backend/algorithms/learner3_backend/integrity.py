"""Class for analysis operations on the scores."""

from __future__ import division
import torch
import math
import time

class Integrity(object):
    def __init__(self, hyper_params):
        self.step_size = Step_size(hyper_params)
        self.hp = hyper_params
        self.prev_score = torch.tensor(self.hp.initial_score, device='cuda')
        self.score = torch.tensor(self.hp.initial_score, device='cuda')
        self.top = torch.tensor(self.hp.initial_score, device='cuda')
        self.value = self.hp.initial_integrity
        self.step = 0  # State
        self.improvement = False
        self.entropy = 0

    def update_state(self):
        self.step +=1

    def reset_state(self):
        self.step = 0
        self.top = torch.tensor(self.hp.initial_score, device='cuda')

    def set_integrity(self):
        """Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.step_size.set_step_size()
        if not self.improved():
            self.reduce_integrity()
            self.improvement = False
            self.step_size.decrease_bin()

        else:  # Improved
            self.maintain_integrity()
            self.improvement = True
            self.step_size.increase_bin()


        if self.improved_abs():
            self.top = self.score

    def improved_abs(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.step>0:
            if self.hp.minimizing:
                return self.score < self.top
            else:
                return self.score > self.top
        else:
            # Improved over the initial score
            return True

    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        if self.step>0:
            self.set_entropy()
            if self.hp.minimizing:
                return self.entropy <= self.hp.min_entropy
            else:
                return self.entropy >= self.hp.min_entropy
        else:
            return True  # Improved over the initial score

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

    def reduce_integrity(self, step_size):
        # Reduce integrity, but not below the minimum allowed level
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
