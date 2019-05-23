"""Base class for elite."""
import torch

class Analysis(object):
    def __init__(self, hp):
        self.hp = hp
        self.replace = False  # Absolute score
        self.improved = False  # Entropic score
        self.score = self.hp.initial_score
        self.top_score = self.hp.initial_score
        self.entropy = 0.

    def analyze(self, score, top_score):
        self.update_state(score, top_score)
        self.improved = self.better_entropy()
        self.replace = self.better_abs()

    def update_state(self, score, top_score):
        self.score = score
        self.top_score = top_score
        self.replace = False
        self.improved = False

    def better_abs(self):
        """Assesses whether a new elite will replace the current one or not."""
        if self.hp.minimizing:
            return self.score < self.top_score
        else:
            return self.score > self.top_score

    def better_entropy(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        self.set_entropy()
        if self.hp.minimizing:
            return self.entropy <= self.hp.min_entropy
        else:
            return self.entropy >= self.hp.min_entropy

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        """
        normal = self.top_score.ne(0)
        i = torch.sub(self.score, self.top_score)
        if normal:
            i = torch.div(i, self.top_score.abs())
        else:
            # Prevent division by zero
            i = torch.div(i, self.hp.epsilon)
        self.entropy = torch.mul(i, 100)

#
