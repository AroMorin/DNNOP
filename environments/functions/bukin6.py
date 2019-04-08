"""Implementation of the Bukin N. 6 function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/bukin6.html
"""
from .function import Function
import torch

class Bukin6(Function):
    def __init__(self, env_params):
        super(Bukin6, self).__init__(env_params)
        self.x = None  # NP array
        self.x_low = [-15, -5]
        self.x_high = [-3, 3]
        self.optimal_x = [-10, 1]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 100
        b = self.x[1]-(0.01*(self.x[0]**2))
        c = torch.sqrt(torch.abs(b))
        d = 0.01*torch.abs(self.x[0]+10)
        return (a*c)+d









        #
