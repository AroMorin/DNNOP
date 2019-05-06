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

    def step(self):
        """Placeholder method for performing an optimization step."""
        pass

    def reset_state(self):
        """Placeholder method in case the solver has need to reset its internal
        state.
        """
        pass

    def print_state(self):
        """Placeholder method for performing an optimization step."""
        pass

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.top_score
        if self.minimizing:
            return best <= self.target
        else:
            return best >= self.target






#
