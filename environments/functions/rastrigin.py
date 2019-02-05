"""Implementation of the Rastrigin function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/rastr.html
"""
from .function import Function
import numpy as np

class Rastrigin(Function):
    def __init__(self, nb_dimensions, plot):
        super().__init__(plot, nb_dimensions)
        self.x = None  # NP array
        self.x1_low = -5.12
        self.x1_high = 5.12
        self.x2_low = -5.12
        self.x2_high = 5.12
        self.optimal_x1 = 0  # Location
        self.optimal_x2 = 0  # Location
        self.optimal_y = 0
        self.resoultion = 512
        self.minimize = True
        self.set_domains()
        self.set_range()

    def get_func(self):
        a = 10*self.nb_dimensions
        b = np.square(self.x)
        c = 10*np.cos(2*np.pi*self.x)
        d = b - c
        e = np.sum(d)
        return a + e

    def get_funcs(x):
        a = 10*2
        b = np.square(x)
        c = 10*np.cos(2*np.pi*x)
        d = b - c
        e = np.sum(d)
        return a + e

    def evaluate(self, x):
        self.x = x
        self.z = self.get_func()

    def step(self):
        pass

    def plot(self, elite, anchors, probes, blends):
        pass








        #
