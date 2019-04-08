"""Base class for anchors."""

import torch
import math

class Anchors(object):
    def __init__(self, hp):
        self.hp = hp
        self.vectors = []
        self.anchors_idxs = []
        self.nb_anchors = 0  # State not hyperparameter
        self.print_distance = True

    def set_anchors(self, vectors, analyzer):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        self.reset_state()
        #idxs = list(range(self.hp.pool_size))
        anchors_idxs = self.set_anchors_idxs(analyzer.sorted_idxs, vectors)
        self.set_vectors(vectors)
        print("Anchors: ", len(self.anchors_idxs))
        print("Anchors idxs: ", self.anchors_idxs)

        #as_ = [self.analyzer.scores[i] for i in self.anchors.anchors_idxs]
        #print("Anchors scores: ", as_)

    def reset_state(self):
        """Resets the class' states."""
        self.vectors = []
        self.anchors_idxs = []
        self.nb_anchors = 0

    def set_anchors_idxs(self, sorted_idxs, vectors):
        """Determines the indices for anchors."""
        for i in sorted_idxs:
            candidate = vectors[i]
            self.admit(candidate, i, vectors)
            if self.nb_anchors == self.hp.nb_anchors:
                break  # Terminate

    def remove_elite(self, idxs):
        """Removes the elite index."""
        idxs.remove(0)

    def admit(self, candidate, candidate_idx, vectors):
        """Determines whether to admit a sample into the anchors list."""
        if self.anchors_idxs:
            if self.accept_candidate(candidate, vectors):
                self.anchors_idxs.append(candidate_idx)
                self.nb_anchors += 1
        else:
            # List is empty, admit
            self.anchors_idxs.append(candidate_idx)
            self.nb_anchors += 1

    def accept_candidate(self, candidate, vectors):
        """Make sure the candidate is far enough from every anchor."""
        for i in self.anchors_idxs:
            anchor = vectors[i]
            distance = self.canberra_distance(candidate, anchor)
            if self.print_distance:
                print(distance)
            if distance.lt(self.hp.min_dist):
                return False
            if not torch.isfinite(distance):
                return False
        return True

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
        return result

    def set_vectors(self, pool):
        """Sets the vectors of the anchors."""
        for i in self.anchors_idxs:
            self.vectors.append(pool[i])








#
