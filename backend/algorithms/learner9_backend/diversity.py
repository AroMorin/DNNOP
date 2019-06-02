"""Base class for frustration"""

import torch

class Diversity(object):
    def __init__(self, hp):
        self.hp = hp
        self.min_distance = 0.
        self.flag = False

    def update_state(self, improved, integrity):
        self.flag = False
        if improved or integrity==0.99:
            self.reset_state()

    def reset_state(self):
        self.min_distance = 0.

    def check(self, elite, candidate):
        d = self.calculate(elite, candidate)
        self.flag = d>=self.min_distance
        self.set_min_distance(d)

    def calculate(self, a, b):
        x = a.sub(b).abs()
        y = torch.add(a.abs(), b.abs())
        f = torch.div(x, y)
        j = torch.masked_select(f, torch.isfinite(f))
        result = j.sum()
        return result.item()

    def set_min_distance(self, d):
        if self.flag:
            self.min_distance = d
        else:
            self.reduce_distance()

    def reduce_distance(self):
        """This method is necessary for stability"""
        a = self.min_distance*0.005
        b = self.min_distance-a
        self.min_distance = max(0., b)



#
