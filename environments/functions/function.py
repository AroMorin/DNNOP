"""Base class for functions"""

import numpy as np
import matplotlib as plt

class Function:
    def ___init___(self):
        self.func = None
        self.nb_dimensions = 2
        self.x_1 = 0
        self.x_2 = 0
        self.y_1 = 0
        self.y_2 = 0
        self.optimal_val = 0
        self.optimal_x = 0
        self.optimal_y = 0

    def construct_base(self):
        pass

    def evaluate(self, position):
        pass
