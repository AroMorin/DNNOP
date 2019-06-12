"""Base class for elite."""
import torch

class Analysis(object):
    def __init__(self, hp):
        self.hp = hp
        self.analysis = ''  # Absolute score
        self.improved = False  # Entropic score
        self.score = self.hp.initial_score
        self.top_score = self.hp.initial_score
        self.entropy = 0.

    def analyze(self, score, top_score):
        self.update_state(score, top_score)
        self.improved = self.better_entropy()
        self.analysis = self.better_abs()

    def update_state(self, score, top_score):
        self.score = score
        self.top_score = top_score
        self.analysis = ''
        self.improved = False

    def better_abs(self):
        """Assesses whether the score is better or not than the previous one."""
        if self.score == self.top_score:
            result = 'same'
            return result
        elif self.hp.minimizing:
            better = self.score < self.top_score
        else:
            better = self.score > self.top_score
        if better:
            result = 'better'
        else:
            result = 'worse'
        return result

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
