"""Apply Perturbation"""

class Perturbation:
    def __init__(self, hp):
        self.hp = hp
        self.search_radius = 0
        self.num_selections = 0
        self.integrity = self.hp.initial_integrity

    def apply(self, vec):
