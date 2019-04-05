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
from .msn_backend.optimizer import Optimizer
from .hyper_parameters import Hyper_Parameters
from .pool import Pool
import time

class MSN(object):
    def __init__(self, models, alg_params):
        print ("Using MSN algorithm")
        alg_params = self.ingest_params(alg_params)
        self.hyper_params = Hyper_Parameters(hyper_params) # Create a hyper parameters object
        self.pool = Pool(models, self.hp) # Create a pool object
        self.pool_size = len(pool)
        self.train_losses = []
        self.test_loss = []
        self.train_accs = []
        self.test_acc = []
        self.correct_test_preds = []
        self.inferences = []
        self.scores = []
        self.optimizer = alg_params["optimizer"]  # Name of optimizer
        self.optim = None  # Optimizer object
        self.set_optim()

    def ingest_params(self, alg_params):
        default_params = {
                            "optimizer": "default"
                            }
        default_params.update(alg_params)
        return default_params

    def set_optim(self):
        """If the user gives an optimizer, then use it. Otherwise, use the
        default MSN optimizer.
        The given optimizer has to contain the required methods for the MSN
        algorithm to function, for example inference().
        """
        assert "optimizer" in self.hyper_params
        if self.optimizer == "default":
            self.optim = Optimizer(self.pool, self.hyper_params)
        else:
            print("Unknown optimizer, exiting!")
            exit()

    def set_environment(self, env):
        """Sets the environment attribute."""
        self.env = env
        assert self.env is not None  # Sanity check
        if self.env.loss:
            self.scoring = "loss"
        if self.env.acc:
            self.scoring = "acc"
        if self.env.score:
            self.scoring = "score"

    def optimize(self):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.inferences = self.optim.inference(self.env)
        if self.scoring == "loss":
            self.optim.calculate_losses(self.inferences, self.env)
        elif self.scoring == "acc":
            self.optim.calculate_correct_predictions(self.inferences, self.env)
        elif self.scoring == "score":
            self.optim.calculate_scores(self.inferences, self.env)
        else:
            self.optim.set_scores(self.inferences)
        self.optim.step()

    def inference(self, env, test=False):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        inferences = []
        with torch.no_grad():
            if test:
                model = self.pool.models[self.pool.anchors.anchors_idxs[0]]
                model.eval()  # Turn on evaluation mode
                inference = model(env.test_data)
                inferences.append(inference)
            else:
                for model in self.pool.models:
                    inference = model(env.observation)
                    inferences.append(inference)
        self.print_inference(inferences)
        return inferences

    def print_inference(self, outputs):
        """Prints the inference of the neural networks. Attempts to extract
        the output items from the tensors.
        """
        if self.print_inferences:
            if len(outputs[0]) == 1:
                x = [a.item() for a in outputs]
            elif len(outputs[0]) == 2:
                x = [[a[0].item(), a[1].item()] for a in outputs]
            else:
                x = [[tensor_.item() for tensor_ in output_] for output_ in outputs]
            print("Inference: ", x)

    def test(self, env):
        """This is a method for testing."""
        self.inferences = self.optim.inference(test=True)
        if env.test_data is not None:
            self.correct_test_preds = self.optim.calculate_correct_predictions(
                                                    self.inferences, test=True)
        else:
            print ("Environment has no test cases!")
            exit()

    def print_test_accuracy(self, env):
        """Prints the accuracy figure for the test/validation case/set."""
        test_size = len(env.test_data)
        correct = self.correct_test_preds
        self.test_acc = 100.*correct/test_size
        if env.loss:
            loss = self.test_loss[0]  # Assuming minizming loss
            loss /= test_size  # Not really sure what this does
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                                    loss, correct, test_size, self.test_acc))
        else:
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)'.format(
                                    correct, test_size, self.test_acc))

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.optim.pool.elite.elite_score
        if self.optim.hp.minimizing:
            return best <= (self.optim.hp.target + self.optim.hp.tolerance)
        else:
            return best >= (self.optim.hp.target - self.optim.hp.tolerance)

    def save_weights(self, path):
        for i, sample in enumerate(self.pool):
            fn = path+"model_"+str(i)+".pth"
            torch.save(sample.state_dict(), fn)
        fn = path+"model_elite.pth"
        torch.save(self.optim.pool.elite.model.state_dict(), fn)












#
