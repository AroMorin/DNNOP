"""A class that defines interfaces to use the SGD solver in PyTorch.
It is expected that the optimizer object will be compatible with the SGD
interface.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from .algorithm import Algorithm

class SGD(Algorithm):
    def __init__(self, model, optimizer):
        """Model is owned by the class, so it is set as a class attribute.
        """
        print("Using SGD algorithm")
        super().__init__(nb_models=1) # SGD optimizes only one model
        self.model = model # Model is set as a class attribute
        if optimizer == None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        else:
            self.optimizer = optimizer
        self.train_loss = ''
        self.test_loss = ''
        self.train_acc = ''
        self.test_acc = ''
        self.correct_test_preds = ''

    def optimize(self, env):
        """I chose not to use local variables.
        """
        self.model.train() #Sets behavior to "training" mode
        self.optimizer.zero_grad()
        predictions = self.model(env.x)
        self.train_loss = F.nll_loss(predictions, env.y)
        self.train_loss.backward()
        self.optimizer.step()

    def test(self, env):
        """Local variables are chosen not to feature here.
        """
        self.model.eval() #Sets behavior to "training" mode
        with torch.no_grad():
            predictions = model(env.x_t)
            self.test_loss = F.nll_loss(predictions, env.y_t, reduction='sum').item()
            # Get the index of the max log-probability, i.e. prediction per each input
            pred = predictions.max(1, keepdim=True)[1]
            # Number of correct predictions
            self.correct_test_preds = pred.eq(env.y_t.view_as(pred)).sum().item()

    def print_test_accuracy(self, env):
        test_size = len(env.x_t)
        self.test_acc = 100.*self.correct_test_preds/test_size
        loss = self.test_loss
        loss /= test_size # Not really sure what this does
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, self.correct_test_preds, test_size, self.test_acc))







#
