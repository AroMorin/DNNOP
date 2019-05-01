"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""
class Evaluator(object):
    def __init__(self):
        self.env = None
        self.scoring = ''
        self.score = None
        # Will only be used if the appropriate score type is selected
        self.train_loss = 1
        self.test_loss = 1
        self.train_acc = 0
        self.test_acc = 0

    def set_environment(self, env):
        """Sets the environment and the scoring attributes."""
        self.env = env
        assert self.env is not None  # Sanity check
        if self.env.loss:
            self.scoring = "loss"
        elif self.env.acc:
            self.scoring = "accuracy"
        elif self.env.score:
            self.scoring = "score"
        elif self.env.error:
            self.scoring = "error"
        else:
            print("Unknown scoring method, exiting!")
            exit()

    def evaluate(self, inference, test=False):
        if test:
            assert self.env.test_data is not None  # Sanity check
        if self.scoring == "loss":
            self.calculate_loss(inference, test)
        elif self.scoring == "accuracy":
            self.calculate_correct_predictions(inference, test, acc=True)
        elif self.scoring == "score" or self.scoring == "error":
            self.calculate_score(inference, test)
        else:
            self.set_score(inference, test)

    def calculate_loss(self, inference, test=False):
        """This method calculates the loss."""
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
        self.score = self.env.evaluate(inference)

    def set_score(self, score):
        self.score = score

    def reset_state(self):
        # Flush values
        self.train_loss = 1
        self.train_acc = 0
        self.test_loss = 1
        self.test_acc = 0







#
