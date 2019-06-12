"""base class for pool
The pool object will contain the models under optimization.
"""
import torch

class Weights(object):
    def __init__(self, greed):
        self.mu = 0.
        self.greedy = greed

    def update(self, analysis, model):
        if analysis == 'better':
            self.reinforce()
        elif analysis == 'worse':
            self.erode()
        elif analysis == 'same':
            self.maintain()
        self.implement(model)

    def reinforce(self):
        self.mu = 0.0004

    def erode(self):
        self.mu = -0.004

    def maintain(self):
        if self.greedy:
            self.mu = -0.004
        else:
            self.mu = 0.

    def implement(self, model):
        high = 1.
        low = 0.
        v = model.fc1.weight[model.ex1, :]
        v.add_(self.mu)
        v.clamp_(low, high)
        print(model.fc1.weight[0:20])
        model.fc1.weight[model.ex1, :] = v

        v = model.fc2.weight[model.ex2, :]
        v.add_(self.mu)
        v.clamp_(low, high)
        model.fc2.weight[model.ex2, :] = v

        v = model.fc3.weight[model.ex3, :]
        v.add_(self.mu)
        v.clamp_(low, high)
        model.fc3.weight[model.ex3, :] = v

        v = model.fc4.weight[:, model.ex3]
        v.add_(self.mu)
        v.clamp_(low, high)
        model.fc4.weight[:, model.ex3] = v
#
