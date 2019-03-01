"""Implementation of the Bukin N. 6 function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/bukin6.html
"""
from .function import Function
import numpy as np

class Bukin6(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-15, -5]
        self.x_high = [-3, 3]
        self.optimal_x = [-10, 1]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 100
        b = self.x[1]-(0.01*np.square(self.x[0]))
        c = np.sqrt(np.abs(b))
        d = 0.01*np.abs(self.x[0]+10)
        return (a*c)+d









        #
