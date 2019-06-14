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
        self.high = 1.
        self.low = 0.
        self.erosion = True

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
        mu = 0.004
        mu = np.random.choice([0., mu], 1, p=[0.5, 0.5])
        self.mu = 0.0004

    def decay(self):
        self.mu = -0.0004

    def maintain(self):
        if self.greedy:
            self.mu = -0.0004
        else:
            self.mu = 0.

    def erode(self, model):
        self.nu = 0.0001
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

    def implement(self, model):
        v = model.fc1.weight[model.ex1, :]
        v_p = v[:, model.ex0]
        v_p.add_(self.mu)
        v_p.clamp_(self.low, self.high)
        v[:, model.ex0] = v_p
        model.fc1.weight[model.ex1, :] = v

        v = model.fc2.weight[model.ex2, :]
        v_p = v[:, model.ex1]
        v_p.add_(self.mu)
        v_p.clamp_(self.low, self.high)
        v[:, model.ex1] = v_p
        model.fc2.weight[model.ex2, :] = v

        v = model.fc3.weight[model.ex3, :]
        v_p = v[:, model.ex2]
        v_p.add_(self.mu)
        v_p.clamp_(self.low, self.high)
        v[:, model.ex2] = v_p
        model.fc3.weight[model.ex3, :] = v

        v = model.fc4.weight[:, model.ex3]
        v.add_(self.mu)
        v.clamp_(self.low, self.high)
        model.fc4.weight[:, model.ex3] = v
        print(model.fc4.weight[0])









#
