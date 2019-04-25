"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

import torch
import torch.nn.functional as F
import time

class Optimizer(object):
    def __init__(self, pool, hyper_params):
        self.pool = pool
        self.hp = hyper_params
        self.env = None
        self.score = self.hp.initial_score
        # Will only be used if the appropriate score type is selected
        self.train_loss = 1
        self.test_loss = 1
        self.train_acc = 0
        self.test_acc = 0

    def set_environment(self, env):
        self.env = env

    def reset_state(self):
        # Flush values
        self.train_loss = 1
        self.train_acc = 0
        self.test_loss = 1
        self.test_acc = 0

    def calculate_loss(self, inference, test=False):
        """This method calculates the loss."""
        self.inference = inference
        if self.env.loss_type == 'NLL loss':
            if not test:
                self.train_loss = F.nll_loss(inference, self.env.labels)
                self.score = self.train_loss
            else:
                loss = F.nll_loss(inference, self.env.test_labels, reduction='sum').item()
                self.test_loss = loss
        else:
            print("Unknown loss type")
            exit()

    def calculate_correct_predictions(self, inference, test=False, acc=False):
        """Calculates the number of correct predictions/inferences made by the
        neural network.
        """
        self.inference = inference
        if not test:
            # Training
            # Correct predictions on all test data for a single model
            pred = inference.argmax(dim=1, keepdim=True)
            correct = pred.eq(self.env.labels.view_as(pred)).sum().float()
            if acc:
                self.abs_to_acc(correct, test=test)
                self.train_acc = correct
            self.score = correct
        else:
            # Testing
            pred = inference.argmax(dim=1, keepdim=True)
            correct = pred.eq(self.env.test_labels.view_as(pred)).sum().float()
            if acc:
                self.abs_to_acc(correct, test=test)
            self.test_acc = correct

    def abs_to_acc(self, a, test):
        """Absolute number to accuracy percentage. These are in-place
        modification/ops on a torch tensor. It is assumed that they translate,
        and thus no need to return the tensor back to the caller func.
        """
        if not test:
            size = len(self.env.observation)
        else:
            size = len(self.env.test_data)
        a.div_(size)
        a.mul_(100)

    def calculate_score(self, inference):
        """Calculates the scores given the network inferences."""
        self.inference = inference
        self.score = self.env.evaluate(inference)

    def set_score(self, score):
        self.score = score

    def step(self):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        print("Score: %f" %self.score.item())
        self.pool.prep_new_model(self.env.observation, self.env.labels, self.inference, self.score)
        self.pool.generate()
        #self.pool.evaluate()
        #self.pool.print_state()



#
