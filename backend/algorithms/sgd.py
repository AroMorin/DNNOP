"""A class that defines interfaces to use the SGD solver in PyTorch.
It is expected that the optimizer object will be compatible with the SGD
interface.
"""

from __future__ import division
import torch
import torch.nn.functional as F
import torch.optim as optim

class SGD(object):
    def __init__(self, model, alg_params):
        """Model is owned by the class, so it is set as a class attribute.
        Think of the optimizer as the "engine" behind the algorithm.
        """
        print("Using SGD algorithm")
        alg_params = elf.ingest_params(alg_params)
        self.hyper_params = alg_params
        self.model = model # Model is set as a class attribute
        self.train_loss = 0.
        self.test_loss = 0.
        self.train_acc = 0.
        self.test_acc = 0.
        # Number of instances where the model's prediction was correct
        self.correct_test_preds = 0
        self.optimizer = alg_params["optimizer"]  # Name of the optimizer
        self.optim = None  # The optimizer object
        self.set_optim()

    def ingest_params(self, alg_params):
        """Method to loop over the hyper parameter dictionary and update the
        default values.
        """
        default_params = {
                            "learning rate": 0.01,
                            "momentum": 0.5,
                            "optimizer": "SGD"
                            }
        default_params.update(alg_params)
        return alg_params

    def set_optim(self, optimizer):
        """Method to set the desired optimizer, using the desired hyper
        parameters.
        """
        if self.optimizer == "SGD":
            self.optim = optim.SGD(self.model.parameters(),
                                        lr = self.hyper_params["learning rate"],
                                        momentum = self.hyper_params["momentum"])
        else:
            print("Unkown optimizer, exiting!")
            exit()

    def set_environment(self, env):
        self.env = env

    def optimize(self, env):
        """Implements the main optimization function of the algorithm."""
        self.set_environment(env)
        self.model.train() #Sets behavior to "training" mode
        self.optimizer.zero_grad()
        predictions = self.model(env.x)
        self.train_loss = F.nll_loss(predictions, env.y)
        self.train_loss.backward()
        self.optimizer.step()

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        if self.env.minimize:
            return self.test_loss <= self.hyper_params["target loss"]
        else:
            return self.test_loss >= self.hyper_params["target loss"]

    def test(self, env):
        """Local variables are chosen not to feature here.
        """
        self.model.eval() #Sets behavior to "training" mode
        with torch.no_grad():
            predictions = self.model(env.x_t)
            self.test_loss = F.nll_loss(predictions, env.y_t, reduction='sum').item()
            # Get the index of the max log-probability, i.e. prediction per each input
            pred = predictions.max(1, keepdim=True)[1]
            # Number of correct predictions
            self.correct_test_preds = pred.eq(env.y_t.view_as(pred)).sum().item()

    def print_test_accuracy(self, env):
        """Prints the accuracy figure for the test/validation case/set."""
        test_size = len(env.x_t)
        self.test_acc = 100.*self.correct_test_preds/test_size
        loss = self.test_loss
        loss /= test_size # Not really sure what this does
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, self.correct_test_preds, test_size, self.test_acc))







#
