"""Base class for anchors"""

import math

class Anchors:
    def __init__(self, hp):
        self.hp = hp
        self.models = []
        self.anchors_idxs = []
        self.elapsed_steps = 0

    def set_anchors(self, pool, scores, elite):
        self.reset_state()
        if not self.backtracking:
            sorted_scores = self.sort_scores(scores)
            sorted_idxs = self.get_sorted_idxs(sorted_scores, scores)
            anchors_idxs = self.get_anchors_idxs(sorted_idxs, pool)
            self.assign_models()
        else:
            self.models[-1] = elite  # Replace worst anchor with Elite
            # Reset state
            self.backtracking = False
            self.elapsed_steps = 0

    def reset_state(self):
        self.models = []
        self.anchors_idxs = []

    def set_anchors_idxs(self, sorted_idxs, pool):
        for i in sorted_idxs:
            candidate = pool[i]
            self.admit(candidate, pool)
            if len(self.anchors_idxs) == self.hp.nb_anchors:
                # Terminate
                break

    def admit(self, candidate, pool):
        """Determines whether to admit a sample into the anchors list."""
        if self.anchors_idxs:
            if self.check_distances(candidate, pool):
                self.anchors_idxs.append(i)
        else:
            # List is empty, admit
            self.anchors_idxs.append(i)

    def check_distances(self, candidate, pool):
        for i in self.anchors_idxs:
            anchor = pool[i]
            distance = self.canberra_distance(candidate, anchor)
            if distance<self.hp.min_dist:
                return False
        return True

    def canberra_distance(a, b):
        """Calculates Canberra distance between two vectors."""
        numerator = torch.abs(torch.add(a, -1, b))
        denominator = torch.add(torch.abs(a), torch.abs(b))
        return torch.div(numerator, denominator)

    def assign_models(self, pool):
        for i in self.anchors_idxs:
            self.models.append(pool[i])









#
