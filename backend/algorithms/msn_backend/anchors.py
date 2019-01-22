"""Base class for anchors"""

class Anchors:
    def __init__(self, hp):
        self.hp = hp
        self.models = []
        self.anchors_idxs = []
        self.backtracking = False
        self.elapsed_steps = 0

    def set_anchors(self, pool, scores, elite):
        self.reset_state()
        self.review()
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

    def review(self):
        if self.elapsed_steps > self.hp.patience:
            self.backtracking = True
        else:
            self.elapsed_steps += 1

    def sort_scores(self, scores):
        """This function sorts the values in the list. Duplicates are removed
        also.
        """
        if self.hp.minimizing:
            sorted_scores = sorted(set(scores))
        else:
            sorted_scores = sorted(set(scores), reverse=True)
        return sorted_scores

    def get_sorted_idxs(self, sorted_scores, scores):
        """This function checks each element in the sorted list to retrieve
        all matching indices in the original scores list. This preserves
        duplicates. It reduces the likelihood that anchor slots will be
        unfilled.
        """
        sorted_idxs = []
        for i in range(len(scores)):
            score = sorted_scores[i]
            idxs = [idx for idx, value in enumerate(scores) if value == score]
            sorted_idxs.append(idxs)
        # Sanity checks
        assert len(sorted_idxs) == len(scores)  # No missing elements
        assert len(set(sorted_idxs)) == len(sorted_idxs)  # No duplicates
        return sorted_idxs

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
