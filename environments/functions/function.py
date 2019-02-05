"""Base class for functions"""

import numpy as np
import matplotlib as plt
from .plotter import Plotter

class Function(Environment):
    def ___init___(self, plot, nb_dimensions):
        self.plot = plot
        self.nb_dimensions = nb_dimensions
        self.optimal_val = 0
        self.optimal_x1 = 0  # Location
        self.optimal_x2 = 0  # Location
        self.optimal_y = 0  # Value
        self.x1_low = 0
        self.x1_high = 0
        self.x2_low = 0
        self.x2_high = 0
        self.y = 0
        self.resoultion = 512
        self.minimize = True  # Global optimum is a minimum/maximum
        self.x1_domain = []
        self.x2_domain = []
        self.domain = []
        self.init_plot()

    def init_plot(self):
        if self.plot:
            assert self.nb_dimensions == 2
            self.plotter = Plotter()

    def set_domains(self):
        x1 = np.linspace(self.x1_low, self.x1_high, self.resolution)
        x2 = np.linspace(self.x2_low, self.x2_high, self.resolution)
        self.domain = np.meshgrid(x1_domain, x2_domain)
        #self.x1_domain, self.x2_domain = np.meshgrid(x1_domain, x2_domain)

    def set_range(self):
        self.x = self.domain
        self.range = self.get_func()

    def construct_base(self):
        pass

    def evaluate(self, position):
        pass
