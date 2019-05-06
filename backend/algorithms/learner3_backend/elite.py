"""Base class for elite."""

class Elite(object):
    def __init__(self, hp):
        self.hp = hp
        self.vector = None
        self.elite_score = hp.initial_score
        self.replaced_elite = False  # State

    def set_elite(self, vector, score):
        """Checks current top score and determines if there's a new elite. The
        elite is then either updated or set as is.
        """
        self.replaced_elite = False  # Operating assumption
        if self.replace(score):
            self.replaced_elite = True
            self.elite_score = score
            self.preserve(vector)

    def replace(self, score):
        """Assesses whether a new elite will replace the current one or not."""
        if self.hp.minimizing:
            return score < self.elite_score
        else:
            return score > self.elite_score

    def preserve(self, vector):
        """We clone the elite in order to have our own copy of it, not just a
        pointer to the object.
        """
        self.vector = vector.clone()



#
