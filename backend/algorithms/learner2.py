"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since Learner is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
from __future__ import division
import torch
import numpy as np
from .learner2_backend.hyper_parameters import Hyper_Parameters
from .learner2_backend.pool import Pool
from .learner2_backend.optimizer import Optimizer
import time

class LEARNER2(object):
    def __init__(self, models, alg_params):
        print ("Using Learner2 algorithm")
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.pool = Pool(models, self.hyper_params) # Create a pool object
        self.optim = Optimizer(self.pool, self.hyper_params)  # Optimizer object
        self.inference = None
        self.correct_test_preds = 0
        self.populations = False

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
        self.get_inference()
        if self.scoring == "loss":
            self.optim.calculate_loss(self.inference)
        elif self.scoring == "accuracy":
            self.optim.calculate_correct_predictions(self.inference, acc=True)
        elif self.scoring == "score" or self.scoring == "error":
            self.optim.calculate_score(self.inference)
        else:
            self.optim.set_score(self.inference)
        self.optim.step()

    def get_inference(self, test=False, silent=True):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        with torch.no_grad():
            if not test:
                # Training
                model = self.pool.model
                self.inference = model(self.env.observation)
            else:
                # Testing
                model = self.pool.elite.model
                model.eval()  # Turn on evaluation mode
                self.inference = model(self.env.test_data)
        if not silent:
            self.print_inference()

    def print_inference(self):
        """Prints the inference of the neural networks. Attempts to extract
        the output items from the tensors.
        """
        if len(self.inference) == 1:
            x = self.inferences.item()
        elif len(self.inference) == 2:
            x = (self.inference[0].item(), self.inference[1].item())
        else:
            x = [a.item() for a in self.inference]
        print("Inference: ", x)

    def test(self):
        """This is a method for testing."""
        assert self.env.test_data is not None  # Sanity check
        self.get_inference(test=True)
        self.optim.calculate_correct_predictions(self.inference, test=True, acc=True)
        if self.env.loss:
            self.optim.calculate_loss(self.inference, test=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_acc = self.optim.test_acc
        if self.env.loss:
            test_loss = self.optim.test_loss  # Assuming minizming loss
            test_loss /= len(self.env.test_data)
            print('Test set: Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(test_loss,
                                                                test_acc))
        else:
            print('Test set: Accuracy: ({:.0f}%)\n'.format(test_acc))

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
        torch.save(self.pool.elite.model.state_dict(), fn)












#
