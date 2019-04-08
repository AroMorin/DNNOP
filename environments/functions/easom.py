"""Implementation of the Easom function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/easom.html
"""
from .function import Function
import math
import torch

class Easom(Function):
    def __init__(self, env_params):
        super(Easom, self).__init__(env_params)
        self.x = None  # NP array
        self.x_low = [-100, -100]
        self.x_high = [100, 100]
        self.optimal_x = [math.pi, math.pi]  # Location
        self.target = -1
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = torch.cos(self.x[0])*torch.cos(self.x[1])
        b = (self.x[0]-math.pi)**2
        c = (self.x[1]-math.pi)**2
        d = -b-c
        e = -a*torch.exp(d)
        return e









        #
