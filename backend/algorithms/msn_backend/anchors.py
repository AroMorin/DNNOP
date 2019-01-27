"""Base class for anchors"""

import torch

class Anchors:
    def __init__(self, hp):
        self.hp = hp
        self.models = []
        self.anchors_idxs = []
        self.nb_anchors = self.hp.nb_anchors  # State not hyperparameter

    def set_anchors(self, pool, analyzer, elite):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        self.reset_state()
        anchors_idxs = self.set_anchors_idxs(analyzer.sorted_idxs, pool)
        self.assign_models(pool)
        if analyzer.backtracking:
            print ("Backtracking Activated! Inserting Elite")
            self.models[-1] = elite  # Insert elite in last position

    def reset_state(self):
        self.models = []
        self.anchors_idxs = []

    def set_anchors_idxs(self, sorted_idxs, pool):
        for i in sorted_idxs:
            candidate = pool[i]
            self.admit(candidate, i, pool)
            if len(self.anchors_idxs) == self.hp.nb_anchors:
                # Terminate
                break

    def admit(self, candidate, candidate_idx, pool):
        """Determines whether to admit a sample into the anchors list."""
        if self.anchors_idxs:
            if self.check_distances(candidate, pool):
                self.anchors_idxs.append(candidate_idx)
        else:
            # List is empty, admit
            self.anchors_idxs.append(candidate_idx)

    def check_distances(self, candidate, pool):
        """Make sure the candidate is far enough from every anchor."""
        for i in self.anchors_idxs:
            anchor = pool[i]
            distance = self.canberra_distance(candidate, anchor)
            if distance.item() < self.hp.min_dist:
                return False
        return True

    def canberra_distance(self, a, b):
        """Calculates Canberra distance between two vectors."""
        #numerator = torch.abs(torch.add(a, -1, b))
        numerator = torch.abs(a.sub(b))
        denominator = torch.add(torch.abs(a), torch.abs(b))
        return torch.div(numerator, denominator).sum()

    def assign_models(self, pool):
        self.models = []  # Reset state
        for i in self.anchors_idxs:
            self.models.append(pool[i])









#
