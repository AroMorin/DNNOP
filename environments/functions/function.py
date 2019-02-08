"""Base class for functions"""

import numpy as np
import matplotlib as plt
from .plotter import Plotter
from ..environment import Environment

class Function(Environment):
    def __init__(self, plot):
        super().__init__()
        self.plot = plot
        self.optimal_x = 0  # Location
        self.resolution = 50
        self.symmetrical = True
        self.x_low = 0
        self.x_high = 0
        self.domain = []  # Matrix of coordinate vectors
        self.range = []  # Matrix of function values

    def init_plot(self):
        if self.plot:
            self.plotter = Plotter(self)

    def set_observation(self):
        self.observation = np.random.uniform(self.x_low, self.x_high, 2)

    def set_domain(self):
        if self.symmetrical:
            x1 = np.linspace(self.x_low, self.x_high, self.resolution)
            x2 = np.linspace(self.x_low, self.x_high, self.resolution)
        else:
            x1 = np.linspace(self.x_low[0], self.x_high[0], self.resolution)
            x2 = np.linspace(self.x_low[1], self.x_high[1], self.resolution)
        m1, m2 = np.meshgrid(x1, x2)
        self.domain = [m1, m2]

    def set_range(self):
        self.x = self.domain
        self.range = self.get_func()

    def construct_base(self):
        pass

    def evaluate(self, position):
        pass



#
