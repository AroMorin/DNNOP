"""A class that defines interfaces to use the SGD solver in PyTorch."""

import torch
from .algorithm import Algorithm

class SGD(Algorithm):
    def __init__(self, model, optimizer):
        super().__init__(model) # Initialize base class
        self.optimizer = optimizer
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
        model = self.model
        optimizer = self.optimizer
        data = env.x
        targets = env.y

        # Process
        model.train() #Sets behavior to "training" mode
        optimizer.zero_grad()
        predictions = self.model(data)
        loss = F.nll_loss(predictions, env.y)
        loss.backward()
        optimizer.step()

        # State update
        self.train_loss = loss

    def test(self, env):
        # Local variable definitions
        model = self.model
        optimizer = self.optimizer
        data = env.x_t
        targets = env.y_t

        # Process
        model.eval() #Sets behavior to "training" mode
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
        test_loss /= test_size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_size, 100.*correct/test_size))

        # State update
        self.test_acc = 100.*correct/nb_images

    def reset_state(self):
        self.test_loss = 0
        self.correct_test_preds = 0








#
