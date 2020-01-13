"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since MSN is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
from __future__ import division
from .algorithm import Algorithm
import torch
import numpy as np
from .msn2_backend.hyper_parameters import Hyper_Parameters
from .msn2_backend.engine import Engine
import time

class MSN2(Algorithm):
    def __init__(self, models, alg_params):
        print ("Using MSN2 algorithm")
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.engine = Engine(models, self.hyper_params)
        self.model = models
        self.populations = True
        self.scores = []
        self.initial_score = self.hyper_params.initial_score
        self.top_score = self.initial_score

    def optimize(self):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.optim.step()

    def print_inferences(self):
        """Prints the inference of the neural networks. Attempts to extract
        the output items from the tensors.
        """
        if len(self.inferences[0]) == 1:
            x = [a.item() for a in self.inferences]
        elif len(self.inferences[0]) == 2:
            x = [[a[0].item(), a[1].item()] for a in self.inferences]
        else:
            x = self.inferences[0:3]
        print("Inference: ", x)

    def test(self):
        """This is a method for testing."""
        assert self.env.test_data is not None  # Sanity check
        self.get_inference(test=True)
        self.optim.calculate_correct_predictions(self.inferences, test=True, acc=True)
        if self.env.loss:
            self.optim.calculate_losses(self.inferences, test=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_acc = self.optim.test_acc
        if self.env.loss:
            test_loss = self.optim.test_loss  # Assuming minizming loss
            test_loss /= len(self.env.test_data)
            print('\nTest set: Loss: {:.4f}, Accuracy: ({:.0f}%)'.format(test_loss,
                                                                test_acc))
        else:
            print('\nTest set: Accuracy: ({:.0f}%)'.format(test_acc))

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.optim.pool.elite.elite_score
        if self.hyper_params.minimizing:
            return best <= (self.hyper_params.target + self.hyper_params.tolerance)
        else:
            return best >= (self.hyper_params.target - self.hyper_params.tolerance)

    def save_weights(self, path):
        for i, sample in enumerate(self.pool.models):
            fn = path+"model_"+str(i)+".pth"
            torch.save(sample.state_dict(), fn)
        fn = path+"model_elite.pth"
        torch.save(self.optim.pool.elite.model.state_dict(), fn)












#
