"""This is the Evaluator class. Its job is to evaluate the inferences and
acquire feedback. It holds the important self.score attribute.
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
        """This is a handler method to activate the proper routine.
        There is no grad flag for populations because gradients are
        only used for single-candidate optimization.
        """
        if type(inference) is not list:
            self.loss_single(env, inference, test, grad)
        else:
            self.loss_populations(env, inference, test)

    def loss_single(self, env, inference, test=False, grad=False):
        """Calculate loss for a single-candidate optimizer."""
        if env.loss_type == 'NLL loss':
            if not test:
                # Training
                self.train_loss = F.nll_loss(inference, env.labels)
                self.score = self.train_loss
            else:
                # Testing
                with torch.no_grad():
                    loss = F.nll_loss(inference, env.test_labels.cuda(),
                                    reduction='sum').item()
                    self.test_loss = loss/env.test_size
        elif env.loss_type == 'CE loss':
            if not test:
                self.train_loss = F.cross_entropy(inference, env.labels)
                self.score = self.train_loss
            else:
                with torch.no_grad():
                    loss = F.cross_entropy(inference, env.test_labels.cuda(),
                                                reduction='sum').item()
                    self.test_loss = (loss/env.test_size)
        else:
            print("Unknown loss type")
            exit()

    def loss_populations(self, env, inferences, test=False):
        """Calculates loss for multi-candidate methods."""
        with torch.no_grad():
            if env.loss_type == 'NLL loss':
                if not test:
                    # Training
                    self.train_loss = [F.nll_loss(infer, env.labels)
                                        for infer in inferences]
                    self.score = self.train_loss
                else:
                    # Testing
                    losses = [F.nll_loss(infer, env.test_labels.cuda(),
                                    reduction='sum').item() for infer in inferences]
                    self.test_loss = losses/env.test_size
            elif env.loss_type == 'CE loss':
                if not test:
                    self.train_loss = [F.cross_entropy(infer, env.labels)
                                        for infer in inferences]
                    self.score = self.train_loss
                else:
                    losses = [F.cross_entropy(infer, env.test_labels.cuda(),
                                reduction='sum').item() for infer in inferences]
                    self.test_loss = (losses/env.test_size)
            else:
                print("Unknown loss type")
                exit()

    def calculate_correct_predictions(self, env, inference, test=False, acc=False):
        """This is a handler method to activate the proper routine.
        The acc flag determines whether to return a percentage accuracy or
        just the absolute number of correct preds.
        """
        if type(inference) is not list:
            self.acc_single(env, inference, test, acc)
        else:
            self.acc_populations(env, inference, test, acc)

    def acc_single(self, env, inference, test=False, acc=False):
        """Calculates the number of correct predictions/inferences made by the
        neural network. This method is for single-candidate algorithms.
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
            correct = pred.eq(env.test_labels.cuda().view_as(pred)).sum().float()
            if acc:
                self.abs_to_acc(env, correct, test=test)
            self.test_acc = correct

    def acc_populations(self, env, inferences, test=False, acc=False):
        """Calculates the number of correct predictions/inferences made by the
        neural network. This method is for populations.
        """
        if not test:
            # Training
            # Correct predictions on all test data for a single model
            preds = [infer.argmax(dim=1, keepdim=True) for infer in inferences]
            corrects = [pred.eq(env.labels.view_as(pred)).sum().float()
                        for pred in preds]
            if acc:
                for correct in corrects:
                    self.abs_to_acc(env, correct, test=test)
                self.train_acc = corrects
            self.score = corrects
        else:
            # Testing
            preds = [infer.argmax(dim=1, keepdim=True) for infer in inferences]
            #pred = inference.argmax(dim=1, keepdim=True)[1]
            corrects = [pred.eq(env.test_labels.cuda().view_as(pred)).sum().float()
                        for pred in preds]
            if acc:
                for correct in corrects:
                    self.abs_to_acc(env, correct, test=test)
            self.test_acc = corrects

    def abs_to_acc(self, env, a, test):
        """Absolute number to accuracy percentage. These are in-place
        modification/ops on a torch tensor. It is assumed that they translate,
        and thus no need to return the tensor back to the caller func.
        """
        if not test:
            size = len(env.observation)
        else:
            size = env.test_size
        a.div_(size)
        a.mul_(100)

    def calculate_score(self, env, inference):
        """This is a handler method to activate the proper routine."""
        if type(inference) is not list:
            self.score_single(env, inference)
        else:
            self.score_populations(env, inference)

    def score_single(self, env, inference):
        """Calculates the scores given the network inference."""
        self.score = torch.Tensor([env.evaluate(inference)])

    def score_populations(self, env, inferences):
        """Calculates the scores given the network inferences."""
        self.score = [torch.Tensor([env.evaluate(infer)]).cuda()
                        for infer in inferences]

    def set_score(self, score):
        self.score = score

    def clean_score(self, env):
        """Removes deformities in the score list such as NaNs."""
        if env.minimize:
            a = float('inf')  # Initial score
        else:
            a = -float('inf')
        if type(self.score) is not list:
            x = self.score
            y = torch.full_like(x, a)
            score = torch.where(torch.isfinite(x), x, y)
            self.score = score.float().cpu()
        else:
            scores = []
            for x in self.score:
                y = torch.full_like(x, a)
                score = torch.where(torch.isfinite(x), x, y)
                score = score.float().cpu()
                scores.append(score)
            self.score = scores

    def reset_state(self):
        # Flush values
        self.train_loss = 10.
        self.train_acc = 0.
        self.test_loss = 10.
        self.test_acc = 0.







#
