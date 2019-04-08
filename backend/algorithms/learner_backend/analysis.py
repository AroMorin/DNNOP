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
        self.search_start = True
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
        x = torch.stack(x)
        temp = torch.zeros_like(x)
        # Removes NaNs and infinities
        self.scores = torch.where(torch.isfinite(x), x, temp)

    def sort_scores(self):
        """This function sorts the values in the list. Duplicates are removed
        also.
        """
        self.current_top = self.new_top  # Inheritance
        if self.hp.minimizing:
            self.sorted = self.scores.sort()
        else:
            self.sorted = self.scores.sort(descending=True)
        # .sort() returns two lists they are assigned below
        self.sorted_scores = self.sorted[0]
        self.sorted_idxs = self.sorted[1]

        # Update state
        self.new_top = self.sorted_scores[0]
        self.top_idx = self.sorted_idxs[0]
        #print("Pool top score: %f" %self.new_top)

    def set_integrity(self):
        """Once an improvement is detected, the flag "reset_integrity" is set
        to True. This means that if later there wasn't an improvement, integrity
        would be reset. Hence, it ensures that integrity restarts with every
        improvement, and only with improvement. If once searching starts, then
        integrity is reduced normally.
        """
        if not self.improved():
            print ("No improvement")
            if not self.search_start:
                # Reduce integrity, but not below the minimum allowed level
                a = self.integrity-self.hp.step_size
                b = self.hp.min_integrity
                self.integrity = max(a, b)
                self.elapsed_steps += 1
            else:
                print("Start searching")
                self.integrity = self.hp.def_integrity  # Reset integrity
                self.search_start = False
        else:  # Improved
            print ("Improved")
            self.elapsed_steps = 0
            self.search_start = True

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
            self.step +=1
            return True

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        The integrity needs to be reset at some point, however. Maybe backtracking
        is just enough for now? I want to reset integrity once no improvement
        was detected (but only the first instance of such occasion).
        """
        #t1 = time.time()
        #torch.cuda.synchronize()
        #print("-----------time %s-------------" %(time.time()-t1))
        eps = self.current_top.gt(0)
        if eps:
            # Percentage change
            _ = self.new_top.sub(self.current_top)
            _ = torch.div(_, self.current_top.abs())
            _ = torch.mul(_, 100)
            self.entropy = _
            #self.entropy = ((self.new_top-self.current_top)/abs(self.current_top))*100
        else:
            # Prevent division by zero
            _ = self.new_top.sub(self.current_top)
            _ = torch.div(_, self.hp.epsilon)
            _ = torch.mul(_, 100)
            self.entropy = _
            #self.entropy = ((self.new_top-self.current_top)/self.hp.epsilon)*100
        #print("Entropy: %f" %self.entropy)


    def review(self, nb_anchors):
        """Implements the backtracking and radial expansion functionalities."""
        self.set_backtracking()
        self.set_radial_expansion(nb_anchors)

    def set_backtracking(self):
        """Only activate backtracking for the current iteration, if the conditions
        are met. Then reset it the following turn(s). If activated, reset
        counter."""
        if self.elapsed_steps > self.hp.patience:
            print ("Waited %d steps" %self.elapsed_steps)
            self.backtracking = True
            self.elapsed_steps = 0
            self.integrity = self.hp.def_integrity  # Reset integrity
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
