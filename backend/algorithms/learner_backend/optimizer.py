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

    def set_environment(self, env):
        self.env = env

    def reset_state(self):
        # Flush values
        self.train_losses = []
        self.test_loss = 1

    def calculate_losses(self, inferences, test=False):
        """This method calculates the loss."""
        if self.env.loss_type == 'NLL loss':
            losses = []
            for idx, inference in enumerate(inferences):
                if not test:
                    loss = F.nll_loss(inference, self.env.labels)
                    self.train_losses.append(loss)
                else:
                    loss = F.nll_loss(inference, self.env.test_labels, reduction='sum').item()
                    self.test_loss = loss
                losses.append(loss)
            self.scores = losses
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
            for idx, inference in enumerate(inferences):
                # Correct predictions on all test data for a single model
                pred = inference.max(1, keepdim=True)[1]
                correct = pred.eq(self.env.labels.view_as(pred)).sum().float()
                if acc:
                    size = len(self.env.observation)
                    correct.div_(size)
                    correct.mul_(100)
                collection.append(correct)
        else:
            # Testing
            pred = inferences.max(1, keepdim=True)[1]
            collection = pred.eq(self.env.test_labels.view_as(pred)).sum().item()
        self.scores = collection

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
