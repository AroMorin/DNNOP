"""Base class for elite."""

import copy

class Elite(object):
    def __init__(self, hp):
        self.model = None
        self.hp = hp
        self.elite_score = hp.initial_score
        self.minimizing = hp.minimizing
        self.inference = None
        self.vector = None
        self.replaced_elite = False  # State

    def set_elite(self, model, vector, inference, score):
        """Checks current top score and determines if there's a new elite. The
        elite is then either updated or set as is.
        """
        self.replaced_elite = False  # Operating assumption
        if self.replace(score):
            self.replaced_elite = True
            self.clone_model(model)
            self.vector = vector.clone()
            self.inference = inference
            self.elite_score = score

    def replace(self, score):
        """Assesses whether a new elite will replace the current one or not."""
        if self.minimizing:
            return score < self.elite_score
        else:
            return score > self.elite_score

    def clone_model(self, model):
        """We clone the elite in order to have our own copy of it, not just a
        pointer to the object. This will be important because we want to
        keep the elite outside of the pool. Only when backtracking do we insert
        the elite into the pool.
        """
        self.model = copy.deepcopy(model)

    def get_elite(self, observation):
        inference = self.model(observation)
        return inference

    def reset_state(self):
        self.elite_score = self.hp.initial_score
