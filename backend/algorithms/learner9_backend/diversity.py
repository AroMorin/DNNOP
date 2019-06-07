"""We should keep trying to do things that are "farther" away from each other.
Thus, the distance metric increases with each "failure".
"""

import torch

class Diversity(object):
    def __init__(self, hp):
        self.hp = hp
        self.min_distance = 0.
        self.flag = False

    def update_state(self, improved, integrity):
        self.flag = False
        if improved:
            self.reset_state()
        else:
            self.increase_distance()

    def reset_state(self):
        self.min_distance = 0.

    def check(self, elite, candidate):
        d = self.calculate(elite, candidate)
        #print(self.min_distance, d)
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
        a = self.min_distance*0.0005
        b = self.min_distance-a
        self.min_distance = max(0., b)

    def increase_distance(self):
        """This method is necessary for stability"""
        self.min_distance += (self.min_distance*0.01)


#
