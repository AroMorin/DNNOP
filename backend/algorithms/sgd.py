"""A class that defines interfaces to use the SGD solver in PyTorch."""

import torch

class SGD(Algorithm):
    def __init__(self, optimizer):
        super().__init__(model) # Initialize base class
        self.optimizer = optimizer
        self.training_loss = ''
        self.test_loss = ''
        self.training_acc = ''
        self.test_acc = ''
        self.correct_test_preds = ''

    def optimize(self, env):
        # Variable definition
        model = self.model
        optimizer = self.optimizer
        data = env.x
        targets = env.y

        model.train() #Sets behavior to "training" mode
        optimizer.zero_grad()
        predictions = self.model(data)
        self.train_loss = F.nll_loss(predictions, env.y)
        self.train_loss.backward()
        optimizer.step()

    def test(self):
        # variable definitions
        model = self.model
        optimizer = self.optimizer
        data = env.x_t
        targets = env.y_t

        model.eval() #Sets behavior to "training" mode
        with torch.no_grad():
            predictions = model(data)
            self.test_loss = F.nll_loss(predictions, targets, reduction='sum').item()
            # get the index of the max log-probability
            pred = predictions.max(1, keepdim=True)[1]
            self.correct_test_preds += pred.eq(targets.view_as(pred)).sum().item()

    def get_test_accuracy(self):
        # Variable Definitions
        test_size = len(env.x_t)
        correct = self.correct_test_preds
        loss = self.test_loss

        test_loss /= test_size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, test_size, 100.*correct/test_size))
        self.test_acc = 100.*correct/nb_images

    def reset_state(self):
        self.test_loss = 0
        self.correct_test_preds = 0








#
