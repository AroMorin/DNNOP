"""A class that defines interfaces to use the SGD solver in PyTorch.
It is expected that the optimizer object will be compatible with the SGD
interface.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

class SGD:
    def __init__(self, model, hyper_params, optimizer):
        """Model is owned by the class, so it is set as a class attribute."""
        print("Using SGD algorithm")
        self.model = model # Model is set as a class attribute
        self.train_loss = ''
        self.test_loss = ''
        self.train_acc = ''
        self.test_acc = ''
        self.correct_test_preds = ''
        self.hyper_params = {}
        self.optimizer = None
        self.set_hyperparams(hyper_params)
        self.set_optimizer(optimizer)

    def set_hyperparams(self, hyper_params):
        """Method to loop over the hyper parameter dictionary and update the
        default values.
        """
        self.hyper_params = {
                            "learning rate": 0.01,
                            "momentum":0.5
                            }
        for key in hyper_params:
            assert key in self.hyper_params
            self.hyper_params[key] = hyper_params[key]

    def set_optimizer(self, optimizer):
        """Method to set the desired optimizer, using the desired hyper
        parameters.
        """
        if optimizer == None:
            self.optimizer = optim.SGD(
                            self.model.parameters(),
                            lr=self.hyper_params['learning rate'],
                            momentum=self.hyper_params['momentum']
                            )
        else:
            self.optimizer = optimizer

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
            predictions = self.model(env.x_t)
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
