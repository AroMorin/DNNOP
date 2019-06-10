"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""
import torch

class Algorithm(object):
    def __init__(self):
        self.none = "hahahaha"
        self.top_score = None
        self.initial_score = None
        self.populations = False
        self.model = None
        self.minimizing = True
        self.target = None
        self.hyper_params = None

    def step(self):
        """Placeholder method for performing an optimization step."""
        pass

    def reset_state(self):
        """Placeholder method in case the solver has need to reset its internal
        state.
        """
        self.top_score = self.initial_score

    def print_state(self):
        """Placeholder method for performing an optimization step."""
        pass

    def set_target(self):
        if self.minimizing:
            self.target = self.hyper_params.target + self.hyper_params.tolerance
        else:
            self.target = self.hyper_params.target - self.hyper_params.tolerance

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        if self.minimizing:
            return self.top_score <= self.target
        else:
            return self.top_score >= self.target






#
