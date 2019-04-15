"""Base class for probes"""

from .perturbation import Perturbation
import torch
from scipy import interpolate
import numpy as np

class Memory(object):
    def __init__(self, hp):
        self.hp = hp
        self.observations = []
        self.inferences = []
        self.scores = []
        self.top = self.hp.initial_score
        self.matrix = None
        self.eval = 0
        self.desirable = False
        self.calls = 0
        self.entropy = 0

    def update_state(self, observation, inference, score):
        self.calls = 0
        if observation not in self.observations:
            self.observations.append(observation)
        # ONLY works for single observations
        self.inferences.append(inference.cpu().numpy())
        self.scores.append(score.cpu().numpy())
        assert len(self.inferences) == len(self.scores)  # Sanity check

    def evaluate_model(self, model):
        self.calls +=1
        if len(self.scores)<4:
            self.desirable = True
            return
        for observation in self.observations:
            with torch.no_grad():
                hypothesis = model(observation)
            self.evaluate_hypothesis(hypothesis)
        print("Expected: %f" %self.eval)

    def evaluate_hypothesis(self, hypothesis):
        x = hypothesis.cpu().numpy()
        xp = np.array(self.inferences)
        fp = np.array(self.scores)
        matrix = interpolate.LinearNDInterpolator(xp, fp, self.hp.initial_score)
        self.eval = matrix(x)
        self.evaluate_attractiveness()

    def evaluate_attractiveness(self):
        self.desirable = False
        if self.hp.minimizing:
            self.top = min(self.scores)
            self.set_entropy()
            if self.entropy<=0.05:
                self.desirable = True
        else:
            self.top = max(self.scores)
            self.set_entropy()
            if -0.05<self.entropy:
                self.desirable = True

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        """
        eps = self.top != 0
        if eps:
            # Percentage change
            i = np.subtract(self.eval, self.top)
            i = np.divide(i, np.abs(self.top))
            i = np.multiply(i, 100)
            self.entropy = i
        else:
            # Prevent division by zero
            i = np.subtract(self.eval, self.top)
            i = np.divide(i, self.hp.epsilon)
            i = np.multiply(i, 100)
            self.entropy = i


#
