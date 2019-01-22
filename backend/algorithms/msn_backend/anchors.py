"""Base class for anchors"""

class Anchors:
    def __init__(self, hp):
        self.nb_anchors = hp.nb_anchors
        self.anchors = []
        self.backtracking = False
        self.minimizing = hp.minimizing

    def review(self):
        if self.elapsed_steps > self.patience:
            self.backtracking = True

    def set_anchors(self, pool, scores, elite):
        self.review()
        if not self.backtracking:
            pass
        else:
            anchors[0] = elite

    def sort_scores(self, scores):
        if
        sorted_scores = scores.sort()
