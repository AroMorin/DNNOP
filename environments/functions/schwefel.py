"""Implementation of the Schwefel function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/schwef.html
"""
from .function import Function
import torch

class Schwefel(Function):
    def __init__(self, env_params):
        super(Schwefel, self).__init__(env_params)
        self.x = None  # NP array
        self.x_low = [-500, -500]
        self.x_high = [500, 500]
        self.optimal_x = [420.9687, 420.9687]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 2*418.9829
        b = self.x[0]*torch.sin(torch.sqrt(torch.abs(self.x[0])))
        c = self.x[1]*torch.sin(torch.sqrt(torch.abs(self.x[1])))
        d = b+c
        e = a-d
        return e









        #
