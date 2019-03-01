"""Implementation of the Ackley function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/ackley.html
"""
from .function import Function
import numpy as np

class Ackley(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-32.768, -32.768]
        self.x_high = [32.768, 32.768]
        self.optimal_x = [0, 0]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 20
        b = 0.2
        c = 2*np.pi
        d = np.sum(np.square(self.x))*0.5
        e = -b*np.sqrt(d)
        f = -a*np.exp(e)
        g = np.sum(np.cos(np.multiply(self.x, c)))
        h = np.exp(0.5*g)
        i = f-h+a+np.exp(1)
        return i









        #
