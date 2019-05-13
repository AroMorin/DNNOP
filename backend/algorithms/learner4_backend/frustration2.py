"""Base class for frustration"""

import math
import numpy as np
from collections import deque, Counter

class Frustration(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.mem_size = hp.mem_size
        self.table = [None]
        self.count = 0
        self.tau = 0.001
        self.jump = False

    def update(self, item):
        item = item.item()
        self.append_table(item)
        self.update_count()
        self.set_tau()
        self.set_value()
        self.set_jump()

    def append_table(self, item):
        if self.table[-1] == item:
            self.table.append(item)
        else:
            self.table = [item]

    def update_count(self):
        self.count = len(self.table)

    def set_tau(self):
        """Return most common n element (n=1) as a list. First entry in list is
        at index 0.
        """
        if self.count > self.mem_size:
            r = self.count/self.mem_size
            self.tau = r-1.
            print(self.tau)
        else:
            self.tau = 0.001

    def set_value(self):
        """Sets the frustration value based on tau. The function has a slow
        slope, and then rises until 0.9. It's a function with attractive
        properties in range[0, 1].
        """
        argument = (4*self.tau)-2.0
        exp1 = math.tanh(argument)+1
        self.value = exp1*0.5

    def set_jump(self):
        """As the frustruation increases, the probability of a "jump" increases
        thus getting unstuck.
        """
        p0 = 1.-self.value
        p1 = self.value
        np.random.seed()
        jump = np.random.choice([0, 1], 1, p=[p0, p1])
        self.jump = bool(jump)
#
