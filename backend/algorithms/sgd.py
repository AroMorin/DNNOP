"""A class that defines interfaces to use the SGD solver in PyTorch.
It is expected that the optimizer object will be compatible with the SGD
interface.
"""

from __future__ import division
from .algorithm import Algorithm

import torch
import torch.nn.functional as F
import torch.optim as optim

class SGD(Algorithm):
    def __init__(self, model, alg_params):
        """Model is owned by the class, so it is set as a class attribute.
        Think of the optimizer as the "engine" behind the algorithm.
        """
        print("Using SGD algorithm")
        super(SGD, self).__init__()
        alg_params = self.ingest_params(alg_params)
        self.hyper_params = alg_params
        self.model = model # Model is set as a class attribute
        self.populations = False
        self.engine = None  # The optimizer object
        self.set_optim(alg_params["optimizer"])

    def ingest_params(self, alg_params):
        """Method to loop over the hyper parameter dictionary and update the
        default values.
        """
        default_params = {
                            "learning rate": 0.01,
                            "momentum": 0.5,
                            "optimizer": "SGD",
                            "target loss": 0.
                            }
        default_params.update(alg_params)
        return default_params

    def set_optim(self, optimizer):
        """Method to set the desired optimizer, using the desired hyper
        parameters.
        """
        if optimizer == "SGD":
            self.optim = optim.SGD(self.model.parameters(),
                                        lr = self.hyper_params["learning rate"],
                                        momentum = self.hyper_params["momentum"])
        else:
            print("Unknown optimizer, exiting!")
            exit()

    def set_environment(self, env):
        self.env = env

    def step(self, feedback):
        """Implements the main optimization function of the algorithm."""
        inference, score = feedback
        score.backward()
        self.optim.step()
        self.optim.zero_grad()

    def test(self):
        """Local variables are chosen not to feature here.
        """
        self.model.eval() #Sets behavior to "training" mode
        with torch.no_grad():
            predictions = self.model(self.env.test_data)
            self.test_loss = F.nll_loss(predictions, self.env.test_labels, reduction='sum').item()
            # Get the index of the max log-probability, i.e. prediction per each input
            pred = predictions.max(1, keepdim=True)[1]
            # Number of correct predictions
            self.correct_test_preds = pred.eq(self.env.test_labels.view_as(pred)).sum().item()

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_size = len(self.env.test_data)
        self.test_acc = 100.*self.correct_test_preds/test_size
        loss = self.test_loss
        loss /= test_size # Not really sure what this does
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, self.correct_test_preds, test_size, self.test_acc))

    def print_state(self):
        """Prints the accuracy figure for the test/validation case/set."""
        train_size = len(self.env.observation)
        pred = self.inference.max(1, keepdim=True)[1]
        # Number of correct predictions
        self.correct_train_preds = pred.eq(self.env.labels.view_as(pred)).sum().item()
        self.train_acc = 100.*self.correct_train_preds/train_size
        loss = self.train_loss
        loss /= train_size # Not really sure what this does
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, self.correct_train_preds, train_size, self.train_acc))






#
