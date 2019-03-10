"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since MSN is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
from __future__ import division
import torch
import numpy as np
from msn_backend.optimizer import Optimizer

class MSN(object):
    def __init__(self, pool, hyper_params, optimizer):
        print ("Using MSN algorithm")
        self.pool = pool
        self.pool_size = len(pool)
        self.optim = optimizer
        self.train_losses = []
        self.test_loss = []
        self.train_accs = []
        self.test_acc = []
        self.correct_test_preds = []
        self.inferences = []
        self.hyper_params = hyper_params
        self.scores = []
        self.set_optimizer()

    def set_optimizer(self):
        """If the user gives an optimizer, then use it. Otherwise, use the
        default MSN optimizer.
        The given optimizer has to contain the required methods for the MSN
        algorithm to function, for example inference().
        """
        if self.optim == None:
            self.optim = Optimizer(self.pool, self.hyper_params)
        else:
            self.optim = self.optim

    def optimize(self, env):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.optim.set_environment(env)  # Candidate for repositioning
        self.inferences = self.optim.inference()
        if env.loss:
            self.scores = self.optim.calculate_losses(self.inferences)
        elif env.acc:
            self.scores = self.optim.calculate_correct_predictions(self.inferences)
        elif env.score:
            self.scores = self.optim.calculate_scores(self.inferences)
        else:
            self.scores = self.inferences
        self.optim.update(self.scores)

    def test(self, env):
        """This is a method for testing."""
        self.inferences = self.optim.inference(test=True)
        if env.test_data is not None:
            self.correct_test_preds = self.optim.calculate_correct_predictions(
                                                    self.inferences, test=True)
        else:
            print ("Environment has no test cases!")
            exit()

    def print_test_accuracy(self, env):
        """Prints the accuracy figure for the test/validation case/set."""
        test_size = len(env.test_data)
        correct = self.correct_test_preds
        self.test_acc = 100.*correct/test_size
        if env.loss:
            loss = self.test_loss[0]  # Assuming minizming loss
            loss /= test_size  # Not really sure what this does
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                                    loss, correct, test_size, self.test_acc))
        else:
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                                    correct, test_size, self.test_acc))

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.optim.pool.elite.elite_score
        if self.optim.hp.minimizing:
            return best <= (self.optim.hp.target + self.optim.hp.tolerance)
        else:
            return best >= (self.optim.hp.target - self.optim.hp.tolerance)














#
