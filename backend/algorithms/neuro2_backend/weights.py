"""base class for pool
The pool object will contain the models under optimization.
"""
import torch
import numpy as np

class Weights(object):
    def __init__(self, hp, greed):
        self.hp = hp
        self.mu = 0.
        self.nu = 0.002
        self.greedy = greed
        self.high = 10.
        self.low = -10.
        self.erosion = False

    def update(self, analysis, model):
        print(analysis)
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
        mu = 0.005
        mu = np.random.choice([0., mu], 1, p=[0.5, 0.5])
        self.mu = mu[0]

    def decay(self):
        mu = -0.005
        mu = np.random.choice([0., mu], 1, p=[0.5, 0.5])
        self.mu = mu[0]

    def maintain(self):
        if self.greedy:
            self.mu = -0.001
            exit()
        else:
            self.mu = 0.

    def erode(self, model):
        self.nu = 0.00001
        v = model.fc1.weight[:, :]
        v.sub_(self.nu)
        model.fc1.weight[:, :] = v

        v = model.fc2.weight[:, :]
        v.sub_(self.nu)
        model.fc2.weight[:, :] = v

        v = model.fc3.weight[:, :]
        v.sub_(self.nu)
        model.fc3.weight[:, :] = v

        v = model.fc4.weight[:, :]
        v.sub_(self.nu)
        model.fc4.weight[:, :] = v

    def implement(self, model):
        v = model.fc1.weight[model.ex1, :]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
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

    def implement_(self, model):
        v = model.fc1.weight[model.ex1, :]
        v_p = v[:, model.ex0]
        v_p.add_(self.mu)
        v[:, model.ex0] = v_p
        v.clamp_(self.low, self.high)
        model.fc1.weight[model.ex1, :] = v

        v = model.fc2.weight[model.ex2, :]
        v_p = v[:, model.ex1]
        v_p.add_(self.mu)
        v[:, model.ex1] = v_p
        v.clamp_(self.low, self.high)
        model.fc2.weight[model.ex2, :] = v

        v = model.fc3.weight[model.ex3, :]
        v_p = v[:, model.ex2]
        v_p.add_(self.mu)
        v[:, model.ex2] = v_p
        v.clamp_(self.low, self.high)
        model.fc3.weight[model.ex3, :] = v

        v = model.fc4.weight[:, model.ex3]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        model.fc4.weight[:, model.ex3] = v






#
