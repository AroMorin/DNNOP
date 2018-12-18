"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
class Algorithm:
    def __init__(self, model):
        self.model = model # A model to optimize
        self.x = '' # Training data
        self.y = '' # Training labels/targets
        self.x_t = '' # Testing data
        self.y_t = '' # Testing labels/targets
        self.optimizer = '' # Optimizer for SGD-derivatives
        self.nb_batches = '' # If data is separated into batches
        self.nb_test_batches = ''
        self.pool_size = '' # Size of pool, for pool-based algorithms


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
