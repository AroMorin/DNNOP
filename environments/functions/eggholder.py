"""Implementation of the Eggholder function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/egg.html
"""
from .function import Function
import numpy as np

class Eggholder(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-512, -512]
        self.x_high = [512, 512]
        self.optimal_x = [512, 404.2319]  # Location
        self.target = -959.6407
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        a = self.x[1]+47
        b = a+(0.5*self.x[0])
        c = np.sin(np.sqrt(np.abs(b)))
        d = self.x[0]-a
        e = np.sin(np.sqrt(np.abs(d)))
        f = (-a*c)-(self.x[0]*e)
        return f









        #
