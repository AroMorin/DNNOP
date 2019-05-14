"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""
import torch
import torch.nn.functional as F

class Evaluator(object):
    def __init__(self):
        # It is assumed we always want to maximize
        self.score = None
        # Will only be used if the appropriate score type is selected
        self.train_loss = 10.
        self.test_loss = 10.
        self.train_acc = 0.
        self.test_acc = 0.

    def evaluate(self, env, inference, test=False):
        if test:
            assert env.test_data is not None  # Sanity check
        if env.score_type == "loss":
            self.calculate_loss(env, inference, test)
        elif env.score_type == "accuracy":
            self.calculate_correct_predictions(env, inference, test, acc=True)
        elif env.score_type == "score" or env.score_type == "error":
            self.calculate_score(env, inference)
        else:
            self.set_score(inference)
        self.clean_score(env)

    def calculate_loss(self, env, inference, test=False, grad=False):
        """This method calculates the loss."""
        if env.loss_type == 'NLL loss':
            if not test:
                self.train_loss = F.nll_loss(inference, env.labels)
                self.score = self.train_loss
            else:
                with torch.no_grad():
                    loss = F.nll_loss(inference, env.test_labels,
                                    reduction='sum').item()
                    self.test_loss = loss
        elif env.loss_type == 'CE loss':
            if not test:
                self.train_loss = F.cross_entropy(inference, env.labels)
                self.score = self.train_loss
            else:
                with torch.no_grad():
                    loss = F.cross_entropy(inference, env.test_labels,
                                                reduction='sum').item()
                    self.test_loss = loss
        else:
            print("Unknown loss type")
            exit()

    def calculate_correct_predictions(self, env, inference, test=False, acc=False):
        """Calculates the number of correct predictions/inferences made by the
        neural network.
        """
        if not test:
            # Training
            # Correct predictions on all test data for a single model
            pred = inference.argmax(dim=1, keepdim=True)
            correct = pred.eq(env.labels.view_as(pred)).sum().float()
            if acc:
                self.abs_to_acc(env, correct, test=test)
                self.train_acc = correct
            self.score = correct
        else:
            # Testing
            pred = inference.argmax(dim=1, keepdim=True)
            #pred = inference.argmax(dim=1, keepdim=True)[1]
            correct = pred.eq(env.test_labels.view_as(pred)).sum().float()
            if acc:
                self.abs_to_acc(env, correct, test=test)
            self.test_acc = correct

    def abs_to_acc(self, env, a, test):
        """Absolute number to accuracy percentage. These are in-place
        modification/ops on a torch tensor. It is assumed that they translate,
        and thus no need to return the tensor back to the caller func.
        """
        if not test:
            size = len(env.observation)
        else:
            size = len(env.test_data)
        a.div_(size)
        a.mul_(100)

    def calculate_score(self, env, inference):
        """Calculates the scores given the network inferences."""
        self.score = env.evaluate(inference)

    def set_score(self, score):
        self.score = score

    def clean_score(self, env):
        """Removes deformities in the score list such as NaNs."""
        if env.minimize:
            a = float('inf')  # Initial score
        else:
            a = -float('inf')
        x = self.score
        y = torch.full_like(x, a)
        self.score = torch.where(torch.isfinite(x), x, y).float()

    def reset_state(self):
        # Flush values
        self.train_loss = 10.
        self.train_acc = 0.
        self.test_loss = 10.
        self.test_acc = 0.







#
