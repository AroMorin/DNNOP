"""Implementation of the Rosenbrock function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/rosen.html
"""
from .function import Function
import numpy as np

class Rosenbrock(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-5, -5]
        self.x_high = [10, 10]
        self.optimal_x = [1, 1]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        a = 100
        b = np.square(self.x[1]-np.square(self.x[0]))
        c = np.square(self.x[0]-1)
        d = (a*b)+c
        return d









        #
