"""Implementation of the Eggholder function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/egg.html
"""
from .function import Function
import torch

class Eggholder(Function):
    def __init__(self, env_params):
        super(Eggholder, self).__init__(env_params)
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
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = self.x[1]+47
        b = a+(0.5*self.x[0])
        c = torch.sin(torch.sqrt(torch.abs(b)))
        d = self.x[0]-a
        e = torch.sin(torch.sqrt(torch.abs(d)))
        f = (-a*c)-(self.x[0]*e)
        return f









        #
