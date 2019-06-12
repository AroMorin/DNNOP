"""base class for pool
The pool object will contain the models under optimization.
"""
import torch
import numpy as np

class Weights(object):
    def __init__(self, greed):
        self.mu = 0.
        self.nu = 0.002
        self.greedy = greed
        self.high = 1.
        self.low = 0.
        self.erosion = True

    def update(self, analysis, model):
        if analysis == 'better':
            self.reinforce()
        elif analysis == 'worse':
            self.decay()
        elif analysis == 'same':
            self.maintain()
        if self.erosion:
            self.erode(model)
        self.implement(model)

    def reinforce(self):
        mu = np.random.choice([0., 0.004], 1, p=[0.5, 0.5])
        self.mu = mu[0]

    def decay(self):
        self.mu = -0.004

    def maintain(self):
        if self.greedy:
            self.mu = -0.004
        else:
            self.mu = 0.

    def implement(self, model):
        v = model.fc1.weight[model.ex1, :]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        print(model.fc1.weight[0:20])
        model.fc1.weight[model.ex1, :] = v

        v = model.fc2.weight[model.ex2, :]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        model.fc2.weight[model.ex2, :] = v

        v = model.fc3.weight[model.ex3, :]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        model.fc3.weight[model.ex3, :] = v

        v = model.fc4.weight[:, model.ex3]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        model.fc4.weight[:, model.ex3] = v

    def erode(self, model):
        self.nu = max(0.1*abs(self.mu), 0.0001)
        v = model.fc1.weight[:, :]
        v.sub_(self.nu)
        v.clamp_(self.low, self.high)
        model.fc1.weight[:, :] = v

        v = model.fc2.weight[:, :]
        v.sub_(self.nu)
        v.clamp_(self.low, self.high)
        model.fc2.weight[:, :] = v

        v = model.fc3.weight[:, :]
        v.sub_(self.nu)
        v.clamp_(self.low, self.high)
        model.fc3.weight[:, :] = v

        v = model.fc4.weight[:, :]
        v.sub_(self.nu)
        v.clamp_(self.low, self.high)
        model.fc4.weight[:, :] = v

#
