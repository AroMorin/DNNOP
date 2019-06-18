"""Base class for elite."""
from .intrinsic_reward import IR

import numpy as np

class Analysis(object):
    def __init__(self, hp):
        self.hp = hp
        self.intrinsic_reward = IR(hp)
        self.analysis = ''  # Absolute score
        self.improved = False  # Entropic score
        self.s0 = self.hp.initial_score
        self.s1 = self.hp.initial_score
        self.entropy = 0.

    def analyze(self, feedback, s0):
        observation, inference, reward = feedback
        self.intrinsic_reward.compute(observation, inference)
        s1 = self.compute_s1(reward)
        self.update_state(s0, s1)
        #self.analysis = self.better_abs()
        self.analysis = self.calc_entropy()

    def compute_s1(self, reward):
        #reward.fill_(0.)
        if self.hp.minimizing:
            return reward-self.intrinsic_reward.value
        else:
            return reward+self.intrinsic_reward.value

    def update_state(self, s0, s1):
        self.s0 = s0
        self.s1 = s1
        self.analysis = ''
        self.improved = False

    def better_abs(self):
        """Assesses whether the score is better or not than the previous one."""
        if self.s1 == self.s0:
            result = 'same'
            return result
        elif self.hp.minimizing:
            better = self.s1 < self.s0
        else:
            better = self.s1 > self.s0
        if better:
            result = 'better'
        else:
            result = 'worse'
        return result

    def calc_entropy(self):
        """Assesses whether the score is better or not than the previous one."""
        self.improved = self.better_entropy()
        if self.improved:
            result = 'better'
        else:
            result = 'worse'
        return result

    def better_entropy(self):
        """Calculate whether the score has satisfactorily improved or not based
        on the pre-defined hyper parameters.
        """
        self.set_entropy()
        if self.hp.minimizing:
            return self.entropy <= self.hp.min_entropy
        else:
            return self.entropy >= self.hp.min_entropy

    def set_entropy(self):
        """Function is constructed such that the conditional will evaluate to
        True most of the time.
        """
        normal = self.s0 != 0.
        i = self.s1 - self.s0
        if normal:
            i = i/abs(self.s0)
        else:
            # Prevent division by zero
            i = i/self.hp.epsilon
        self.entropy = i*100.

#
