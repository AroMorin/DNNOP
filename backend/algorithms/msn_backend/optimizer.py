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
        self.integrity = self.hp.initial_integrity
        self.scores = []
        # Will only be used if the appropriate score type is selected
        self.train_losses = []
        self.test_loss = 1
        self.test_acc = 0

    def set_environment(self, env):
        self.env = env

    def reset_state(self):
        # Flush values
        self.train_losses = []
        self.test_loss = 1
        self.test_acc = 0

    def calculate_losses(self, inferences, test=False):
        """This method calculates the loss."""
        if self.env.loss_type == 'NLL loss':
            losses = []
            if not test:
                for inference in inferences:
                    loss = F.nll_loss(inference, self.env.labels)
                    self.train_losses.append(loss)
                    losses.append(loss)
                self.scores = losses
            else:
                loss = F.nll_loss(inferences, self.env.test_labels, reduction='sum').item()
                self.test_loss = loss
        else:
            print("Unknown loss type")
            exit()

    def calculate_correct_predictions(self, inferences, test=False, acc=False):
        """Calculates the number of correct predictions/inferences made by the
        neural network.
        """
        if not test:
            # Training
            collection = []
            for inference in inferences:
                # Correct predictions on all test data for a single model
                pred = inference.argmax(dim=1, keepdim=True)
                correct = pred.eq(self.env.labels.view_as(pred)).sum().float()
                if acc:
                    self.abs_to_acc(correct, test=test)
                collection.append(correct)
            self.scores = collection
        else:
            # Testing
            pred = inferences.argmax(dim=1, keepdim=True)
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

    def calculate_scores(self, inferences):
        """Calculates the scores given the network inferences."""
        inferences = torch.stack(inferences)
        scores = self.env.evaluate(inferences)
        self.scores = scores

    def set_scores(self, scores):
        self.scores = scores

    def step(self):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        self.pool.prep_new_pool(self.scores)
        self.pool.implement()




#
