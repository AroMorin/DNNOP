"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since MSN is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
from __future__ import division
import torch
import numpy as np
from .msn2_backend.optimizer import Optimizer
from .msn2_backend.hyper_parameters import Hyper_Parameters
from .msn2_backend.pool import Pool
import time

class MSN2(object):
    def __init__(self, models, alg_params):
        print ("Using MSN2 algorithm")
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.pool = Pool(models, self.hyper_params) # Create a pool object
        self.optim = Optimizer(self.pool, self.hyper_params)
        self.correct_test_preds = 0
        self.inferences = []
        self.scores = []

    def set_environment(self, env):
        """Sets the environment attribute."""
        self.env = env
        assert self.env is not None  # Sanity check
        if self.env.loss:
            self.scoring = "loss"
        if self.env.acc:
            self.scoring = "accuracy"
        if self.env.score:
            self.scoring = "score"
        if self.env.error:
            self.scoring = "error"
        self.optim.set_environment(env)

    def optimize(self):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.inference()
        self.optim.reset_state()
        if self.scoring == "loss":
            self.optim.calculate_losses(self.inferences)
        elif self.scoring == "accuracy":
            self.optim.calculate_correct_predictions(self.inferences, acc=True)
        elif self.scoring == "score" or self.scoring == "error":
            self.optim.calculate_scores(self.inferences)
        else:
            self.optim.set_scores(self.inferences)
        self.optim.step()

    def inference(self, test=False, silent=True):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        with torch.no_grad():
            if not test:
                inferences = []
                for model in self.pool.models:
                    inference = model(self.env.observation)
                    inferences.append(inference)
            else:
                model = self.pool.elite.model
                model.eval()  # Turn on evaluation mode
                inferences = model(self.env.test_data)
        self.inferences = inferences  # Update state
        if not silent:
            self.print_inferences()

    def print_inferences(self):
        """Prints the inference of the neural networks. Attempts to extract
        the output items from the tensors.
        """
        if len(self.inferences[0]) == 1:
            x = [a.item() for a in self.inferences]
        elif len(self.inferences[0]) == 2:
            x = [[a[0].item(), a[1].item()] for a in self.inferences]
        else:
            x = self.inferences[0:3]
        print("Inference: ", x)

    def test(self):
        """This is a method for testing."""
        assert self.env.test_data is not None  # Sanity check
        self.inference(test=True)
        self.optim.calculate_correct_predictions(self.inferences, test=True, acc=True)
        if self.env.loss:
            self.optim.calculate_losses(self.inferences, test=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_acc = self.optim.test_acc
        if self.env.loss:
            test_loss = self.optim.test_loss  # Assuming minizming loss
            test_loss /= len(self.env.test_data)
            print('\nTest set: Loss: {:.4f}, Accuracy: ({:.0f}%)'.format(test_loss,
                                                                test_acc))
        else:
            print('\nTest set: Accuracy: ({:.0f}%)'.format(test_acc))

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.optim.pool.elite.elite_score
        if self.hyper_params.minimizing:
            return best <= (self.hyper_params.target + self.hyper_params.tolerance)
        else:
            return best >= (self.hyper_params.target - self.hyper_params.tolerance)

    def save_weights(self, path):
        for i, sample in enumerate(self.pool.models):
            fn = path+"model_"+str(i)+".pth"
            torch.save(sample.state_dict(), fn)
        fn = path+"model_elite.pth"
        torch.save(self.optim.pool.elite.model.state_dict(), fn)












#
