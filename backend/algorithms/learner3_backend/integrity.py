"""Class for analysis operations on the scores."""

from __future__ import division
import torch
import math
import time

class Integrity(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.prev_score = torch.tensor(self.hp.initial_score, device='cuda')
        self.score = torch.tensor(self.hp.initial_score, device='cuda')
        self.distance = float("inf")  # Infinite distance from target
        self.top = torch.tensor(self.hp.initial_score, device='cuda')
        self.backtracking = False
        self.elapsed_steps = 0  # Counts steps without improvement
        self.reset_integrity = False
        self.value = self.hp.initial_integrity
        self.step = 0  # State
        self.improvement = False
        self.step_size = self.hp.step_size
        self.bin = [1., 1., 1., 1.]  # Uniform distribution
        self.entropy = 0

    def update_state(self):
        self.step +=1

    def reset_state(self):
        self.step = 0
        self.top = torch.tensor(self.hp.initial_score, device='cuda')

    def set_integrity(self):
        """Once an improvement is detected, the flag "reset_integrity" is set
        to True. This means that if later there wasn't an improvement, integrity
        would be reset. Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.set_step_size()
        if not self.improved():
            self.reduce_integrity()
            self.elapsed_steps += 1
            self.improvement = False
            self.decrease_bin()

        else:  # Improved
            print ("Improved")
            self.increase_bin()
            self.improvement = True
            self.elapsed_steps = 0

        if self.improved_abs():
            self.top = self.score

    def set_step_size(self):
        if  0. < self.value <= 0.25:
            self.step_size = self.hp.step_size/self.bin[0]
        elif  0.25 < self.value <= 0.5:
            self.step_size = self.hp.step_size/self.bin[1]
        elif  0.5 < self.value <= 0.75:
            self.step_size = self.hp.step_size/self.bin[2]
        elif  0.75 < self.value <= 1.:
            self.step_size = self.hp.step_size/self.bin[3]

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

    def reduce_integrity(self):
        # Reduce integrity, but not below the minimum allowed level
        a = self.value-self.step_size  # Decrease integrity
        if a <= self.hp.min_integrity:
            self.value = self.hp.max_integrity  # Trigger backtracking!
        else:
            self.value = max(0, a)  # Integrity never below zero

    def increase_bin(self):
        if  0. < self.value <= 0.25:
            inc = 1./self.bin[0]
            self.bin[0] +=inc
        elif  0.25 < self.value <= 0.5:
            inc = 1./self.bin[1]
            self.bin[1] +=inc
        elif  0.5 < self.value <= 0.75:
            inc = 1./self.bin[2]
            self.bin[2] +=inc
        elif  0.75 < self.value <= 1.:
            inc = 1./self.bin[3]
            self.bin[3] +=inc

    def decrease_bin(self):
        if  0. < self.value <= 0.25:
            self.bin[0] = max(1., self.bin[0]-0.02)
        elif  0.25 < self.value <= 0.5:
            self.bin[1] = max(1., self.bin[1]-0.002)
        elif  0.5 < self.value <= 0.75:
            self.bin[2] = max(1., self.bin[2]-0.002)
        elif  0.75 < self.value <= 1.:
            self.bin[3] = max(1., self.bin[3]-0.02)

    def maintain_integrity(self):
        a = self.value+(self.step_size*2.5)
        b = self.hp.max_integrity
        self.value = min(a, b)

    def set_backtracking(self, trigger=False):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter."""
        if self.step > self.hp.patience or trigger:
            self.backtracking = True
            self.step = 0
            self.value = self.hp.max_integrity  # Reset integrity
        else:
            self.backtracking = False

    def suspend_reality(self):
        self.real_integrity = self.value
        self.real_score = self.score
        self.real_bin = self.bin
        self.real_elapsed_steps = self.elapsed_steps

    def restore_reality(self):
        self.value = self.real_integrity
        self.score = self.real_score
        self.bin = self.real_bin
        self.elapsed_steps = self.real_elapsed_steps
