"""A class that defines interfaces to use the SGD solver in PyTorch."""

import torch
import torch.nn.functional as F
import torch.optim as optim
from .algorithm import Algorithm

class SGD(Algorithm):
    def __init__(self, model):
        print("Using SGD algorithm")
        super().__init__(nb_models=1) # Initialize base class
        self.model = model # A model to optimize
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        self.training_loss = ''
        self.test_loss = ''
        self.training_acc = ''
        self.test_acc = ''
        self.correct_test_preds = ''

    def optimize(self, env):
        """I chose to use local variables to improve readability of the code. If
        at a later stage I find that this affects performance, I can always revert
        back to using class attributes.
        """
        # Local variable definition
        data = env.x
        targets = env.y

        # Process
        self.model.train() #Sets behavior to "training" mode
        self.optimizer.zero_grad()
        predictions = self.model(data)
        loss = F.nll_loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

        # State update
        self.train_loss = loss

    def test(self, env):
        """Local variables are chosen here, this choice is reversible and class
        attributes can be used instead.
        """
        # Local variable definitions
        data = env.x_t
        targets = env.y_t

        # Process
        self.model.eval() #Sets behavior to "training" mode
        with torch.no_grad():
            predictions = model(data)
            loss = F.nll_loss(predictions, targets, reduction='sum').item()
            # get the index of the max log-probability
            pred = predictions.max(1, keepdim=True)[1]
            correct = pred.eq(targets.view_as(pred)).sum().item()

        # State update
        self.test_loss = loss
        self.correct_test_preds = correct

    def get_test_accuracy(self, env):
        # Local variable definitions
        test_size = len(env.x_t)
        correct = self.correct_test_preds
        loss = self.test_loss

        # Process
        loss /= test_size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, correct, test_size, 100.*correct/test_size))

        # State update
        self.test_acc = 100.*correct/test_size

    def reset_state(self):
        self.test_loss = 0
        self.correct_test_preds = 0








#
