"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since MSN is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
import torch
from .msn_backend import optimizer as optim

class MSN:
    def __init__(self, pool, hyper_params, optimizer):
        print ("Using MSN algorithm")
        self.pool_size = len(pool)
        self.optim = None
        self.scores = []
        self.set_optimizer(optimizer)

    def set_optimizer(self, optimizer):
        """If the user gives an optimizer, then use it. Otherwise, use the
        default MSN optimizer.
        The given optimizer has to contain the required methods for the MSN
        algorithm to function, for example inference().
        """
        if optimizer == None:
            self.optim = optimizer(pool, hyper_params)
        else:
            self.optim = optimizer

    def optimize(self, env):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        outputs = self.optim.inference(env)
        self.scores = self.optim.calculate_scores(outputs)
        self.optim.update(self.scores)

    def test(self, env):
        """This is a method for testing."""
        pass

    def achieved_target(self):
        if self.optim.hp["minimization mode"]:
            return self.test_loss <= self.optim.hp["target loss"]
        else:
            return self.test_loss >= self.optim.hp["target loss"]
