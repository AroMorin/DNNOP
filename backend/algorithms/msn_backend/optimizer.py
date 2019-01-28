"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

from .hyper_parameters import Hyper_Parameters
from .pool import Pool
import torch
import torch.nn.functional as F

class Optimizer:
    def __init__(self, models, hyper_params):
        self.hp = Hyper_Parameters(hyper_params) # Create a hyper parameters object
        self.pool = Pool(models, self.hp) # Create a pool object
        self.integrity = self.hp.initial_integrity
        self.env = None

    def set_environment(self, env):
        self.env = env

    def inference(self, test=False):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        assert self.env != None  # Sanity check
        outputs = []
        with torch.no_grad():
            if test:
                model = self.pool.models[0]
                model.eval()  # Turn on evaluation mode
                inf = model(self.env.x_t)  # env.x_t is the test data
                outputs.append(inf)
            else:
                for model in self.pool.models:
                    inf = model(self.env.x)  # env.x is the training data
                    outputs.append(inf)
        return outputs

    def calculate_loss(self, inferences, test=False):
        """This method calculates the loss."""
        if self.env.loss_type == 'NLL loss':
            losses = []
            for inf in inferences:
                if not test:
                    loss = F.nll_loss(inf, self.env.y)
                else:
                    loss = F.nll_loss(inf, self.env.y_t, reduction='sum').item()
                losses.append(loss)
            return losses
        else:
            print("Unknown loss type")
            exit()

    def calculate_scores(self, outputs):
        """This method calculates the scores based on the given outputs. There
        are many ways to calculate a score, it depends on the type of problem
        being solved.
        Thus, this method can use a second argument, or a hyper parameter, to
        set what type of calculation to use.
        """
        scores = []
        for output in outputs:
            score = output-self.hp.target
            scores.append(score)
        return scores

    def update(self, scores):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        self.pool.prep_new_pool(scores)
        self.pool.update_models()


    def calculate_correct_predictions(self, inferences, losses):
        correct_preds = []
        for inference in inferences:
            pred = inference.max(1, keepdim=True)[1]
            # Correct for single model on all test data
            correct = pred.eq(self.env.y_t.view_as(pred)).sum().item()
            correct_preds.append(correct)
        return correct_preds



#
