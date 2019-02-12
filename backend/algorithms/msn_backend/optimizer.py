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
        assert self.env is not None  # Sanity check
        outputs = []
        with torch.no_grad():
            if test:
                model = self.pool.models[self.pool.anchors.anchors_idxs[0]]
                model.eval()  # Turn on evaluation mode
                inference = model(self.env.test_data)
                outputs.append(inference)
            else:
                for model in self.pool.models:
                    inference = model(self.env.observation)
                    outputs.append(inference)
        #self.print_inference(outputs)
        return outputs

    def print_inference(self, outputs):
        print(outputs)
        if outputs[0][0][0].item():
            x = [i.item() for i in outputs[0][0]]
        else:
            x = [i for i in outputs[0][0]]
        print("Inference: ", x)


    def calculate_losses(self, inferences, test=False):
        """This method calculates the loss."""
        if self.env.loss_type == 'NLL loss':
            losses = []
            for inf in inferences:
                if not test:
                    loss = F.nll_loss(inf, self.env.labels)
                else:
                    loss = F.nll_loss(inf, self.env.test_labels, reduction='sum').item()
                losses.append(loss)
            return losses
        else:
            print("Unknown loss type")
            exit()

    def calculate_correct_predictions(self, inferences, test=False):
        correct_preds = []
        for inference in inferences:
            # Correct predictions on all test data for a single model
            pred = inference.max(1, keepdim=True)[1]
            if not test:
                correct = pred.eq(self.env.labels.view_as(pred)).sum().item()
                correct_preds.append(correct)
            else:
                print(len(inferences))
                correct = pred.eq(self.env.test_labels.view_as(pred)).sum().item()
                return correct
        return correct_preds

    def calculate_scores(self, inferences):
        scores = []
        for inference in inferences:
            score = self.env.evaluate(inference)
            scores.append(score)
        return scores

    def update(self, scores):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        self.pool.prep_new_pool(scores)
        self.pool.implement()



#
