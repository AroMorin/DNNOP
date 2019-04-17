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
        self.sims = []
        self.eval = 0
        self.desirable = False
        self.calls = 0
        self.entropy = 0
        self.error = 0

    def update_state(self, observation, inference, score, label=None):
        self.calls = 0
        inference, score = self.reshape(inference, score)
        if label is not None:
            self.label_grouping( observation, inference, score, label)
        else:
            self.observation_grouping()
        self.evaluate_sim()

    def label_grouping(self, observation, inference, score, label):
        if observation not in self.observations:
            # Lists of Tensors
            self.observations.append(observation)
            self.inferences.append(inference)
            self.scores.append(score)
        else:
            for i in range(len(self.observations)):
                if self.observations[i] == observation:
                    self.inferences[i] = torch.cat((self.inferences[i], inference))
                    self.scores[i] = torch.cat((self.scores[i], score))
                    break

    def observation_grouping(self):
        if observation not in self.observations:
            # Lists of Tensors
            self.observations.append(observation)
            self.inferences.append(inference)
            self.scores.append(score)
        else:
            for i in range(len(self.observations)):
                if self.observations[i] == observation:
                    self.inferences[i] = torch.cat((self.inferences[i], inference))
                    self.scores[i] = torch.cat((self.scores[i], score))
                    break


    def reshape(self, a, b):
        if len(inference.shape) == 1:
            inference = inference.reshape(1, inference.shape[0])
        score = score.reshape(1,)

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
        self.evaluate_attractiveness()

    def evaluate_sim(self):
        expected = self.eval
        actual = self.scores[-1]
        self.error = np.abs((expected-actual)/(expected))*100
        if self.error > 0.1:  # 10% error
            self.construct_sim()  # Reconstruct sim

    def construct_sim(self):
        xp = self.inferences.cpu().numpy()
        fp = self.scores.cpu().numpy()
        if self.hp.minimizing:
            opt = self.top-(3*self.top)
        else:
            opt = self.top+(3*self.top)
        self.sim[i] = interpolate.LinearNDInterpolator(xp, fp, opt)

    def evaluate_hypothesis(self, hypothesis, i):
        x = hypothesis.reshape(1,2).cpu().numpy()
        self.eval += self.sim[i](x)

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
