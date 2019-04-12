"""Base class for anchors."""

import torch
import math
import numpy
import copy

class Anchors(object):
    def __init__(self, hp):
        self.hp = hp
        self.vectors = []
        self.scores = []
        self.inferences = []
        self.nb_anchors = 0  # State not hyperparameter
        self.candidates = []
        self.print_distance = True
        self.idxs = []
        self.replace = False
        self.better_score = False
        self.far_enough = False

    def set_anchors(self, vector, inference, score):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        self.reset_state()
        self.check(vector, inference, score)
        self.nb_anchors = len(self.vectors)
        # Sanity checks
        assert len(self.vectors) == len(self.scores)
        assert len(self.vectors) == len(self.inferences)
        assert len(self.vectors) <= self.hp.nb_anchors
        self.sort_anchors()
        print("Anchors: %d" %self.nb_anchors)
        print("Anchors scores: ", self.scores)

    def reset_state(self):
        """Resets the class' states."""
        self.idxs = []
        self.replace = False
        self.better_score = False
        self.far_enough = False

    def check(self, candidate, inference, score):
        """Determines whether to admit a sample into the anchors list."""
        if self.nb_anchors>0:
            self.evaluate_candidate(candidate, score)
            if self.better_score:
                self.admit(candidate, inference, score)
            else:
                if self.nb_anchors<self.hp.nb_anchors:
                    if self.far_enough:
                        print("--Filling up anchor slots--")
                        self.add_anchor(candidate, inference, score)
        else:
            # List is empty, admit
            self.add_anchor(candidate, inference, score)

    def evaluate_candidate(self, candidate, score):
        """Make sure the candidate is far enough from every anchor."""
        # Must be better than current anchor(s) score(s)
        self.evalutate_distance(candidate)
        self.evaluate_score(score)

    def evaluate_score(self, score):
        self.better_score = False  # Starting assumption
        for i, anc_score in enumerate(self.scores):
            if self.hp.minimizing:
                yes = score < anc_score
            else:
                yes = score > anc_score
            if yes:
                self.idxs.append(i)  # Add to list of replacement candidates
                self.better_score = True

    def evalutate_distance(self, candidate):
        distances = []
        self.far_enough = True  # Starting assumption
        self.closest = 0
        c = float("inf")
        #self.candidates = [self.vectors[i] for i in self.idxs]
        for i, anchor in enumerate(self.vectors):
            d = self.canberra_distance(candidate, anchor)
            if d.lt(self.hp.min_dist) and not torch.isfinite(d):
                self.far_enough = False
            if d.lt(c):
                self.closest = i
                c = d  # Update state

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

    def admit(self, vector, inference, score):
        if self.far_enough:
            if self.nb_anchors==self.hp.nb_anchors:
                # Replace anchor at self.idxs[0] is arbitrarily chosen
                self.replace_anchor(vector, inference, score, self.idxs[0])
            else:
                self.add_anchor(vector, inference, score)
                # Fill up anchor slots, anchor is far enough
        else:
            # Replace closest anchor to the candidate
            self.replace_anchor(vector, inference, score, self.idxs[self.closest])
        assert self.nb_anchors <= self.hp.nb_anchors  # Sanity check

    def replace_anchor(self, vector, inference, score, i):
        print("replaced anchor")
        self.vectors[i] = vector.clone()
        self.inferences[i] = inference
        self.scores[i] = score

    def add_anchor(self, vector, inference, score):
        print("added anchor")
        self.vectors.append(vector.clone())
        self.inferences.append(inference)
        self.scores.append(score)

    def sort_anchors(self):
        if self.nb_anchors>1:
            scores = torch.tensor(self.scores, dtype=torch.float)
            print(scores)
            if self.hp.minimizing:
                scores, idxs = scores.sort()
            else:
                scores, idxs = scores.sort(descending=True)
            self.scores = [i.item() for i in scores]
            self.vectors = [self.vectors[i.item()] for i in idxs]
            self.inferences = [self.inferences[i.item()] for i in idxs]
#
