"""Base class for anchors"""

import torch
import math

class Anchors:
    def __init__(self, hp):
        self.hp = hp
        self.models = []
        self.anchors_idxs = []
        self.nb_anchors = 0  # State not hyperparameter

    def set_anchors(self, pool, analyzer):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        self.reset_state()
        anchors_idxs = self.set_anchors_idxs(analyzer.sorted_idxs, pool)
        self.assign_models(pool)
        print("Anchors: ", len(self.anchors_idxs))
        print("Anchors idxs: ", self.anchors_idxs)

        #as_ = [self.analyzer.scores[i] for i in self.anchors.anchors_idxs]
        #print("Anchors scores: ", as_)

    def reset_state(self):
        self.models = []
        self.anchors_idxs = []
        self.nb_anchors = 0

    def set_anchors_idxs(self, sorted_idxs, pool):
        self.remove_elite(sorted_idxs)
        assert 0 not in sorted_idxs  # Sanity check
        for i in sorted_idxs:
            print("anchors ", self.anchors_idxs)
            print("checking %d" %i)
            candidate = pool[i]
            self.admit(candidate, i, pool)
            if self.nb_anchors == self.hp.nb_anchors:
                break  # Terminate

    def remove_elite(self, idxs):
        if 0 in idxs:
            idxs.remove(0)

    def admit(self, candidate, candidate_idx, pool):
        """Determines whether to admit a sample into the anchors list."""
        if self.anchors_idxs:
            if self.accept_candidate(candidate, pool):
                self.anchors_idxs.append(candidate_idx)
                self.nb_anchors += 1
        else:
            # List is empty, admit
            self.anchors_idxs.append(candidate_idx)
            self.nb_anchors += 1

    def accept_candidate(self, candidate, pool):
        """Make sure the candidate is far enough from every anchor."""
        for i in self.anchors_idxs:
            anchor = pool[i]
            distance = self.canberra_distance(candidate, anchor)
            if distance.item() < self.hp.min_dist:
                return False
            elif math.isnan(distance.item()):
                return False
        return True

    def canberra_distance(self, a, b):
        """Calculates Canberra distance between two vectors. There is a bug
        here because the operation is numerically unstable. I think the thing
        is going too fast, perhaps.
        """
        #numerator = torch.abs(torch.add(a, -1, b))
        x = torch.add(a, -b)
        numerator = torch.abs(x)
        y = torch.abs(a)
        z = torch.abs(b)
        deno = torch.add(y, z)
        epsilon = 0.00000001
        denominator = torch.clamp(deno, min=epsilon)
        f = torch.div(numerator, denominator)
        result = f.sum()
        return result

    def assign_models(self, pool):
        for i in self.anchors_idxs:
            self.models.append(pool[i])








#
