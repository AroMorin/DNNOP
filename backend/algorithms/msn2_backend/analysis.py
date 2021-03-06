"""Class for analysis operations on the scores."""

from __future__ import division
import torch
import math
import time

class Analysis(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.current_top = torch.tensor(self.hp.initial_score, device='cuda')
        self.new_top = torch.tensor(self.hp.initial_score, device='cuda')
        self.current_d = float("inf")  # Infinite distance from target
        self.vanilla = torch.full((self.hp.pool_size,), self.hp.initial_score, device='cuda')
        self.top_idx = 0
        self.scores = []
        self.sorted_scores = []
        self.sorted_idxs = []
        self.backtracking = False
        self.elapsed_steps = 0  # Counts steps without improvement
        self.reset_integrity = False
        self.integrity = self.hp.initial_integrity
        self.lr = self.hp.lr
        self.alpha = self.hp.alpha
        self.lambda_ = self.hp.lambda_
        self.search_start = False
        self.nb_anchors = self.hp.nb_anchors  # State
        self.radial_expansion = False
        self.step = 0  # State

    def analyze(self, scores, nb_anchors):
        """The main function."""
        self.clean_list(scores)
        self.sort_scores()
        #print("Sorted scores: ", self.sorted_scores)
        self.set_integrity()
        self.review(nb_anchors)
        self.set_num_selections()
        self.set_search_radius()
        print("Integrity: %f" %self.integrity)

    def clean_list(self, x):
        """Removes deformities in the score list such as NaNs."""
        x = torch.stack(x).float()
        # Removes NaNs and infinities
        self.scores = torch.where(torch.isfinite(x), x, self.vanilla)

    def sort_scores(self):
        """This function sorts the values in the list. Duplicates are removed
        also.
        """
        self.current_top = self.new_top  # Inheritance
        if self.hp.minimizing:
            self.sorted_scores, self.sorted_idxs = self.scores.sort()
        else:
            self.sorted_scores, self.sorted_idxs = self.scores.sort(descending=True)
        # .sort() returns two lists they are assigned below
        self.new_top = self.sorted_scores[0]
        self.top_idx = self.sorted_idxs[0]
        print("Pool top score: %f" %self.new_top)

    def set_integrity(self):
        """Once an improvement is detected, the flag "reset_integrity" is set
        to True. This means that if later there wasn't an improvement, integrity
        would be reset. Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        if not self.improved():
            print ("No improvement")
            # Reduce integrity, but not below the minimum allowed level
            a = self.integrity-self.hp.step_size  # Decrease integrity
            if a <= self.hp.min_integrity:
                self.elapsed_steps = self.hp.patience  # Trigger backtracking!
            else:
                self.integrity = a
            self.elapsed_steps += 1
            self.search_start = True

        else:  # Improved
            print ("Improved")
            if self.search_start:
                if self.integrity<self.hp.def_integrity:
                    print("Reseting Integrity!!!!")
                    self.integrity = self.hp.def_integrity
                self.search_start = False
                self.elapsed_steps += 1
            else:
                # Increase integrity, but not over the maximum allowed level
                self.elapsed_steps = 0
                a = self.integrity+(self.hp.step_size*0.01)
                b = self.hp.max_integrity
                self.integrity = min(a, b)
        print("Steps to Backtrack: %d" %(self.hp.patience-self.elapsed_steps))

    def improved(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        # Make sure we are not in the very first iteration
        if self.step>0:
            new_d = abs(self.new_top-self.hp.target)  # Distance to target
            res = new_d < self.current_d
            self.current_d = new_d
            return res
        else:
            # Improved over the initial score
            self.step +=1
            return True

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
