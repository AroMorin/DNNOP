"""Implementation of the Easom function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/easom.html
"""
from .function import Function
import numpy as np

class Easom(Function):
    def __init__(self, plot, precision, data_path):
        super().__init__(plot, precision)
        self.x = None  # NP array
        self.x_low = [-100, -100]
        self.x_high = [100, 100]
        self.optimal_x = [np.pi, np.pi]  # Location
        self.target = -1
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(data_path)

    def get_func(self):
        a = np.cos(self.x[0])*np.cos(self.x[1])
        b = np.square(self.x[0]-np.pi)
        c = np.square(self.x[1]-np.pi)
        d = -b-c
        e = -a*np.exp(d)
        return e









        #
