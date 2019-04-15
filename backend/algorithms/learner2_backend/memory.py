"""Base class for probes"""

from .perturbation import Perturbation
import torch
from scipy import interpolate
import numpy as np

class Memory(object):
    def __init__(self, hp):
        self.models = []
        self.evals = []
        self.hp = hp
        self.observations = []
        self.inferences = []
        self.scores = []
        self.matrix = None
        self.eval = 0
        self.desirable = False
        self.calls = 0

    def update_state(self, observation, inference, score):
        self.calls = 0
        if observation not in self.observations:
            self.observations.append(observation)
        # ONLY works for single observations
        self.inferences.append(inference.cpu().numpy())
        self.scores.append(score.cpu().numpy())
        assert len(self.inferences) == len(self.scores)  # Sanity check

    def evaluate_model(self, model):
        self.eval = 0  # Reset state
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
        self.matrix = interpolate.LinearNDInterpolator(xp, fp, self.hp.initial_score)
        self.eval += self.matrix(x)
        self.desirable = self.evaluate_attractiveness()

    def evaluate_attractiveness(self):
        if self.hp.minimizing:
            top = min(self.scores)
            return self.eval<top
        else:
            top = max(self.scores)
            return self.eval>top
#
