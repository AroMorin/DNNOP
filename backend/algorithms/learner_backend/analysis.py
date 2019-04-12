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
        self.backtracking = False
        self.elapsed_steps = 0  # Counts steps without improvement
        self.reset_integrity = False
        self.integrity = self.hp.initial_integrity
        self.lr = self.hp.lr
        self.alpha = self.hp.alpha
        self.lambda_ = self.hp.lambda_
        self.search_start = False
        self.nb_anchors = 0  # State
        self.radial_expansion = False
        self.step = 0  # State

    def analyze(self, score, nb_anchors):
        """The main function."""
        self.clean_score(score)
        self.set_integrity()
        self.review(nb_anchors)
        self.set_num_selections()
        self.set_search_radius()
        print("Integrity: %f" %self.integrity)

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
        if not self.improved():
            print ("No improvement")
            self.reduce_integrity()
            self.elapsed_steps += 1
            self.search_start = True

        else:  # Improved
            print ("Improved")
            if self.search_start:
                # Prevents moonshots from disturbing the search process
                if self.integrity<self.hp.def_integrity:
                    print("Reseting Integrity!!!!")
                    self.integrity = self.hp.def_integrity
                self.search_start = False
                self.elapsed_steps += 1
            else:
                # Increase integrity, but not over the maximum allowed level
                self.elapsed_steps = 0
                self.maintain_integrity()
        print("Steps to Backtrack: %d" %(self.hp.patience-self.elapsed_steps+1))

    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.step>0:
            new_d = abs(self.score-self.hp.target)  # Distance to target
            improvement = new_d < self.distance
            if improvement:
                self.distance = new_d  # Update state
            return improvement
        else:
            # Improved over the initial score
            self.step +=1
            return True

    def reduce_integrity(self):
        # Reduce integrity, but not below the minimum allowed level
        a = self.integrity-self.hp.step_size  # Decrease integrity
        if a <= self.hp.min_integrity:
            self.elapsed_steps = self.hp.patience  # Trigger backtracking!
        else:
            self.integrity = min(0, a)  # Integrity never below zero

    def maintain_integrity(self):
        a = self.integrity+(self.hp.step_size*0.1)
        b = self.hp.max_integrity
        self.integrity = min(a, b)

    def review(self, nb_anchors):
        """Implements the backtracking and radial expansion functionalities."""
        self.set_backtracking()
        self.set_radial_expansion(nb_anchors)

    def set_backtracking(self):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter."""
        if self.elapsed_steps > self.hp.patience:
            self.backtracking = True
            self.elapsed_steps = 0
            self.integrity = self.hp.max_integrity  # Reset integrity
        else:
            self.backtracking = False

    def set_radial_expansion(self, nb_anchors):
        """Triggers radial expansion only if the condition is met. Then in the
        new turn it switches it off again.
        It compares the actual number of anchors with the desired number of
        anchors
        """
        if nb_anchors<self.hp.nb_anchors and self.elapsed_steps>0 and nb_anchors<=self.nb_anchors:
            print("--Expanding Search Radius!--")
            self.radial_expansion = True
            self.lr += self.lr * self.hp.expansion_factor
            self.alpha += self.alpha * self.hp.expansion_factor
            self.lambda_ += self.lambda_ * self.hp.expansion_factor
        else:
            self.radial_expansion = False
            self.lr = self.hp.lr
            self.alpha = self.hp.alpha
            self.lambda_ = self.hp.lambda_
        self.nb_anchors = nb_anchors  # State update

    def set_num_selections(self):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        p = 1-self.integrity
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
