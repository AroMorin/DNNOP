"""Implementation of the Schwefel function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/schwef.html
"""
from .function import Function
import numpy as np

class Schwefel(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-500, -500]
        self.x_high = [500, 500]
        self.optimal_x = [420.9687, 420.9687]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 2*418.9829
        b = np.sin(np.sqrt(np.abs(self.x)))
        c = np.add(np.multiply(self.x, b))
        d = a-c
        return d









        #
