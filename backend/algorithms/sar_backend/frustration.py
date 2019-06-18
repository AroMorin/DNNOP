"""Base class for frustration"""

import math
import numpy as np
from collections import deque, Counter

class Frustration(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.mem_size = hp.mem_size
        self.count = 0
        self.tau = 0.00001
        self.jump = False
        self.limit = 0.1

    def update(self, analysis):
        self.update_count(analysis)
        self.set_tau()
        self.set_value()
        if self.count > self.mem_size:
            self.set_jump()
        else:
            self.jump = False

    def update_count(self, analysis):
        if analysis == 'better':
            self.count = 0
        else:
            self.count+=1
        if self.count>1000:
            self.count = 0 # Reset

    def set_tau(self):
        """Return most common n element (n=1) as a list. First entry in list is
        at index 0.
        """
        r = (self.count)/self.mem_size
        self.tau = r-1.

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
        jump = np.random.choice([0, 1], 1, p=[p0, p1])
        self.jump = bool(jump)  # Convert float to boolean

    def reset_state(self):
        self.count = 0


#
