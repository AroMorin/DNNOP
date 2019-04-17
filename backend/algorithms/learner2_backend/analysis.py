"""Class for analysis operations on the scores."""

from __future__ import division
import torch
import math
import time

class Analysis(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.prev_score = torch.tensor(self.hp.initial_score, device='cuda')
        self.score = torch.tensor(self.hp.initial_score, device='cuda')
        self.distance = float("inf")  # Infinite distance from target
        self.top = self.hp.initial_score
        self.backtracking = False
        self.elapsed_steps = 0  # Counts steps without improvement
        self.reset_integrity = False
        self.integrity = self.hp.initial_integrity
        self.lr = self.hp.lr
        self.alpha = self.hp.alpha
        self.lambda_ = self.hp.lambda_
        self.search_start = False
        self.step = -1  # State
        self.improvement = False
        self.step_size = self.hp.step_size
        self.bin = [1., 1., 1., 1.]  # Uniform distribution

    def analyze(self, score):
        """The main function."""
        self.update_state()
        self.clean_score(score)
        self.set_integrity()
        self.review()
        self.set_num_selections()
        self.set_search_radius()
        print("Integrity: %f" %self.integrity)

    def update_state(self):
        self.step +=1
        if self.step > 2500:
            self.bin = [1., 1., 1., 1.]  # Reset bin count
            self.step = 0

    def clean_score(self, x):
        """Removes deformities in the score list such as NaNs."""
        # Removes NaNs and infinities
        y = torch.full_like(x, self.hp.initial_score)
        self.score = torch.where(torch.isfinite(x), x, y)

    def set_integrity(self):
        """Once an improvement is detected, the flag "reset_integrity" is set
        to True. This means that if later there wasn't an improvement, integrity
        would be reset. Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        self.set_step_size()
        if not self.improved():
            print ("No improvement")
            self.reduce_integrity()
            self.elapsed_steps += 1
            self.improvement = False
            self.decrease_bin()

        else:  # Improved
            print ("Improved")
            self.increase_bin()
            self.improvement = True
            self.top = self.score
            self.elapsed_steps = 0
            #self.maintain_integrity()
        print("Steps to Backtrack: %d" %(self.hp.patience-self.elapsed_steps+1))
        print(self.bin)
        print(self.step_size)

    def improved_abs(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.step>0:
            if self.hp.minimizing:
                return self.score <= self.top
            else:
                return self.score >= self.top
        else:
            # Improved over the initial score
            return True

    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.step>0:
            self.set_entropy()
            print("Entropy :%f" %self.entropy)
            if self.hp.minimizing:
                return self.entropy <= self.hp.min_entropy
            else:
                return self.entropy >= self.hp.min_entropy
        else:
            # Improved over the initial score
            return True

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        """
        eps = self.top.ne(0)
        if eps:
            # Percentage change
            i = self.score.sub(self.top)
            i = torch.div(i, self.top.abs())
            i = torch.mul(i, 100)
            self.entropy = i
        else:
            # Prevent division by zero
            i = self.score.sub(self.top)
            i = torch.div(i, self.hp.epsilon)
            i = torch.mul(i, 100)
            self.entropy = i

    def increase_bin(self):
        if  0. < self.integrity <= 0.25:
            self.bin[0] +=1.
        elif  0.25 < self.integrity <= 0.5:
            self.bin[1] +=1.
        elif  0.5 < self.integrity <= 0.75:
            self.bin[2] +=1.
        elif  0.75 < self.integrity <= 1.:
            self.bin[3] +=1.

    def decrease_bin(self):
        if  0. < self.integrity <= 0.25:
            self.bin[0] = max(1., self.bin[0]-0.1)
        elif  0.25 < self.integrity <= 0.5:
            self.bin[1] = max(1., self.bin[1]-0.1)
        elif  0.5 < self.integrity <= 0.75:
            self.bin[2] = max(1., self.bin[2]-0.1)
        elif  0.75 < self.integrity <= 1.:
            self.bin[3] = max(1., self.bin[3]-0.1)

    def set_step_size(self):
        if  0. < self.integrity <= 0.25:
            self.step_size = self.hp.step_size/self.bin[0]
        elif  0.25 < self.integrity <= 0.5:
            self.step_size = self.hp.step_size/self.bin[1]
        elif  0.5 < self.integrity <= 0.75:
            self.step_size = self.hp.step_size/self.bin[2]
        elif  0.75 < self.integrity <= 1.:
            self.step_size = self.hp.step_size/self.bin[3]

    def reduce_integrity(self):
        # Reduce integrity, but not below the minimum allowed level
        a = self.integrity-self.step_size  # Decrease integrity
        if a <= self.hp.min_integrity:
            self.integrity = self.hp.max_integrity  # Trigger backtracking!
        else:
            self.integrity = max(0, a)  # Integrity never below zero

    def maintain_integrity(self):
        a = self.integrity+(self.step_size*2.5)
        b = self.hp.max_integrity
        self.integrity = min(a, b)

    def review(self):
        """Implements the backtracking and radial expansion functionalities."""
        self.set_backtracking()

    def set_backtracking(self, trigger=False):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter."""
        if self.elapsed_steps > self.hp.patience or trigger:
            self.backtracking = True
            self.elapsed_steps = 0
            self.integrity = self.hp.max_integrity  # Reset integrity
        else:
            self.backtracking = False

    def set_num_selections(self):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        #p = 1-self.integrity
        p = self.integrity
        numerator = self.hp.alpha
        denominator = 1+(self.hp.beta/p)
        self.num_selections = numerator/denominator
        print("Num Selections: %f" %self.num_selections)

    def set_search_radius(self):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = 1-self.integrity
        argument = (self.lambda_*p)-2.5
        exp1 = math.tanh(argument)+1
        self.search_radius = exp1*self.lr
        print ("Search Radius: %f" %self.search_radius)














#
