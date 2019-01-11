"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""

class Algorithm:
    def __init__(self, params):
        self.params = params

    def optimize(self):
        """Placeholder method for performing an optimization step."""
        pass

    def test(self):
        """Placeholder method to perform tests/validation.
        """
        pass

    def reset_state(self):
        """Placeholder method in case the solver has need to reset its internal
        state.
        """
        pass
