"""Base class for anchors"""

class Anchors:
    def __init__(self, hp):
        self.hp = hp
        self.models = []
        self.anchors_idxs = []

    def set_anchors(self, pool, analysis, elite):
        """The func is structured as below in order for the conditional to
        evaluate to True most of the time.
        """
        if not analysis.backtracking:
            self.reset_state()
            anchors_idxs = self.get_anchors_idxs(analysis.sorted_idxs, pool)
            self.assign_models()
        else:
            print ("Backtracking Activated! Inserting Elite")
            self.models[-1] = elite  # Insert elite

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
        """Make sure the candidate is far enough from every anchor."""
        for i in self.anchors_idxs:
            anchor = pool[i]
            distance = self.canberra_distance(candidate, anchor)
            if distance < self.hp.min_dist:
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
