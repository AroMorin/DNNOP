"""Base class for probes"""

import torch
from scipy import interpolate
import numpy as np

class Memory(object):
    def __init__(self, hp):
        self.hp = hp
        self.observations = []
        self.concepts = []
        self.inferences = []
        self.scores = []
        self.tops = []
        self.sims = []
        self.evals = []
        self.desirable = False
        self.calls = 0
        self.entropies = []
        self.errors = []

    def init_memory(self, concepts):
        for concept in concepts:
            self.concepts.append(concept)

    def update_state(self, observation, label, inference, score):
        inference, score = self.reshape(inference, score)
        self.observation_grouping(observation, label, inference, score)
        self.evaluate_sim()

    def reshape(self, a, b):
        if len(inference.shape) == 1:
            inference = inference.reshape(1, inference.shape[0])
        score = score.reshape(1,)

    def observation_grouping(self, observation, label, inference, score):
        if label not in self.concepts:
            # First entries for this concept (marked by index)
            self.concepts.append(label.clone())
            self.observations.append([observation.clone()])
            self.inferences.append([inference.clone()])
            self.scores.append([score.clone()])
            self.tops.append(self.hp.initial_score)
            self.evals.append(self.hp.initial_score)
            self.construct_new_sim()
        else:
            # Known concept
            for i in range(len(self.concepts)):
                if self.concepts[i] == label:
                    if not self.tensor_in_list(observation, self.observations[i]):
                        self.observations[i].append(observation)
                    self.inferences[i] = torch.cat((self.inferences[i], inference))
                    self.scores[i] = torch.cat((self.scores[i], score))
                    break

    def new_sim(self):
        xp = self.inferences[-1].cpu().numpy()
        fp = self.scores[-1].cpu().numpy()
        if self.hp.minimizing:
            opt = self.tops[-1]-(3*self.top[-1])
        else:
            opt = self.tops[-1]+(3*self.tops[-1])
        self.sims.append(interpolate.LinearNDInterpolator(xp, fp, opt))

    def tensor_in_list(self, mytensor, mylist):
        for tensor in mylist:
            yes = torch.all(torch.eq(mytensor, tensor))
            if yes:
                return True
        return False

    def evaluate_sim(self, i):
        expected = self.eval
        actual = self.scores[-1]
        self.error = np.abs((expected-actual)/(expected))*100
        if self.error > 0.1:  # 10% error
            self.reconstruct_sim(i)  # Reconstruct sim

    def reconstruct_sim(self, i):
        xp = self.inferences[i].cpu().numpy()
        fp = self.scores[i].cpu().numpy()
        if self.hp.minimizing:
            opt = self.tops[i]-(3*self.tops[i])
        else:
            opt = self.tops[i]+(3*self.tops[i])
        self.sims[i] = interpolate.LinearNDInterpolator(xp, fp, opt)

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
