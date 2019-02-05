"""Base class for functions"""

import numpy as np
import matplotlib as plt
from .plotter import Plotter

class Function(Environment):
    def ___init___(self, nb_dimensions, plot=False):
        super().__init__()
        self.plot = plot
        self.nb_dimensions = nb_dimensions
        self.optimal_x = 0  # Location
        self.resoultion = 50
        self.symmetrical = True
        self.x_low = 0
        self.x_high = 0
        self.domain = []  # Matrix of coordinate vectors
        self.init_plot()

    def init_plot(self):
        if self.plot:
            assert self.nb_dimensions == 2
            self.plotter = Plotter()

    def set_observation(self):
        self.observation = np.rando

    def set_domain(self):
        if self.symmetrical:
            x = []  # List of coordinate vectors
            for _ in range(self.nb_dimensions):
                xi = np.linspace(self.x_low, self.x_high, self.resolution)
                x.append(xi)
        else:
            assert self.nb_dimensions == 2
            x1 = np.linspace(self.x_low[0], self.x_high[0], self.resolution)
            x = [x1, x2]
            x2 = np.linspace(self.x_low[1], self.x_high[1], self.resolution)
        self.domain = np.meshgrid(x)

    def set_range(self):
        self.x = self.domain
        self.range = self.get_func()

    def construct_base(self):
        pass

    def evaluate(self, position):
        pass



#
