"""Base class for probes"""

from .perturbation import Perturbation
import torch
from scipy import interpolate
import numpy as np

class Memory(object):
    def __init__(self, hp):
        self.hp = hp
        self.observations = []
        self.inferences = None
        self.scores = None
        self.top = self.hp.initial_score
        self.matrix = None
        self.eval = 0
        self.desirable = False
        self.calls = 0
        self.entropy = 0

    def update_state(self, observation, inference, score):
        self.calls = 0
        inference = inference.reshape(1, inference.shape[0])
        score = score.reshape(1,)
        if observation not in self.observations:
            self.observations.append(observation)
        # ONLY works for single observations
        #self.inferences.append(inference)
        if type(self.inferences) is not torch.Tensor:
            self.inferences = inference
            self.scores = score
            self.top = score  # Improved over initial score
        else:
            self.inferences = torch.cat((self.inferences, inference))
            self.scores = torch.cat((self.scores, score))

    def evaluate_model(self, model):
        self.calls +=1
        data_points = self.inferences.shape[0]
        if data_points < 4:
            self.desirable = True
            return
        for observation in self.observations:
            with torch.no_grad():
                hypothesis = model(observation)
            self.evaluate_hypothesis(hypothesis)

    def evaluate_hypothesis(self, hypothesis):
        x = hypothesis.reshape(1,2).cpu().numpy()
        xp = self.inferences.cpu().numpy()
        fp = self.scores.cpu().numpy()
        if self.hp.minimizing:
            opt = self.top-(3*self.top)
        else:
            opt = self.top+(3*self.top)
        matrix = interpolate.LinearNDInterpolator(xp, fp, opt)
        self.eval = matrix(x)
        self.evaluate_attractiveness()

    def evaluate_attractiveness(self):
        self.desirable = False
        if self.hp.minimizing:
            self.top = min(self.scores).cpu().numpy()
            self.set_entropy()
            if self.entropy<=-0.1:
                self.desirable = True
        else:
            self.top = max(self.scores).cpu().numpy()
            self.set_entropy()
            if self.entropy>=0.1:
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
