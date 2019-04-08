"""Implementation of the Rosenbrock function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/rosen.html
"""
from .function import Function
import numpy as np

class Rosenbrock(Function):
    def __init__(self, env_params):
        super(Rosenbrock, self).__init__(env_params)
        self.x = None  # NP array
        self.x_low = [-5, -5]
        self.x_high = [10, 10]
        self.optimal_x = [1, 1]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 100
        b = (self.x[1]**2)-(self.x[0]**2)
        #b = np.square(self.x[1]-np.square(self.x[0]))
        c = (self.x[0]**2)-1
        #c = np.square(self.x[0]-1)
        d = (a*b)+c
        return d









        #
