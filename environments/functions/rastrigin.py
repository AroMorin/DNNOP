"""Implementation of the Rastrigin function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/rastr.html
"""
from .function import Function
from .plotter import Plotter
import numpy as np

class Rastrigin(Function):
    def __init__(self, nb_dimensions, plot):
        self.nb_dimensions = nb_dimensions
        self.plot = plot
        self.x = None  # NP array
        self.x1_low = -5.12
        self.x1_high = 5.12
        self.x2_low = -5.12
        self.x2_high = 5.12
        self.y = 0
        self.resoultion = 512
        self.range = set_range()
        self.init_plot()

    def set_range(self):
        x1 = np.linspace(self.x1_low, self.x1_high, self.resolution)
        x2 = np.linspace(self.x2_low, self.x2_high, self.resolution)
        x1_domain, x2_domain = np.meshgrid(x1_domain, x2_domain)
        self.range = self.

    def init_plot(self):
        if self.plot:
            self.plotter = Plotter()

    def get_func(self):
        a = 10*self.nb_dimensions
        b = np.square(self.x)
        c = 10*np.cosine(2*np.pi*self.x)
        d = b - c
        e = np.sum(d)
        return a + e

    def get_func(x):
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
