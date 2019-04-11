"""Base class for anchors."""

import torch
import math
import copy

class Anchors(object):
    def __init__(self, hp):
        self.hp = hp
        self.vectors = []
        self.scores = []
        self.nb_anchors = 0  # State not hyperparameter
        self.print_distance = True
        self.idx = 0
        self.replace = False
        self.better_score = False

    def set_anchors(self, vector, score):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        self.reset_state()
        self.check(vector, score)
        self.nb_anchors = len(self.vectors)
        print("Anchors: %d" %self.nb_anchors)
        #print("Anchors scores: ", self.scores)

    def reset_state(self):
        """Resets the class' states."""
        self.idx = 0
        self.replace = False
        self.better_score = False

    def check(self, candidate, score):
        """Determines whether to admit a sample into the anchors list."""
        if self.nb_anchors>0:
            self.evaluate_candidate(candidate, score)
            if self.replace:
                self.admit(candidate, score)
                self.admitted = True
        else:
            # List is empty, admit
            self.vectors.append(candidate.clone())
            self.scores.append(score)
            self.admitted = True
        # Sanity checks
        assert len(self.vectors) == len(self.scores)
        assert len(self.vectors) <= self.hp.nb_anchors

    def evaluate_candidate(self, candidate, score):
        """Make sure the candidate is far enough from every anchor."""
        # Must be better than current anchor(s) score(s)
        self.evaluate_score(score)
        if self.better_score:
            # Must be far enough from all/the other anchor(s)
            if self.far_enough(candidate):
                self.replace = True

    def evaluate_score(self, score):
        for i, anc_score in enumerate(self.scores):
            if self.hp.minimizing:
                yes = score < anc_score
            else:
                yes = score > anc_score
            if yes:
                self.idx = i  # Add to list of replacement candidates
                self.better_score = True

    def far_enough(self, candidate):
        for anchor in self.vectors:
            distance = self.canberra_distance(candidate, anchor)
            if distance.gt(self.hp.min_dist) and torch.isfinite(distance):
                return False  # Not satisfying distance conditions
        return True # Satisfies conditions

    def canberra_distance(self, a, b):
        """Calculates Canberra distance between two vectors. There is a bug
        here because the operation is numerically unstable. I think the thing
        is going too fast, perhaps.
        """
        x = a.sub(b).abs()
        y = torch.add(a.abs(), b.abs())
        f = torch.div(x, y)
        j = torch.masked_select(f, torch.isfinite(f))
        result = j.sum()
        if self.print_distance:
            print(result)
        return result

    def admit(self, vector, score):
        if self.nb_anchors==self.hp.nb_anchors:
            self.vectors[self.idx] = vector.clone()
            self.scores[self.idx] = score
        else:
            self.vectors.append(vector.clone())
            self.scores.append(score)
        assert self.nb_anchors <= self.hp.nb_anchors  # Sanity check






#
