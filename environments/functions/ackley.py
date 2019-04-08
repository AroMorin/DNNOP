"""Implementation of the Ackley function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/ackley.html
"""
from .function import Function
import math
import torch

class Ackley(Function):
    def __init__(self, env_params):
        super(Ackley, self).__init__(env_params)
        self.x = None  # NP array
        self.x_low = [-32.768, -32.768]
        self.x_high = [32.768, 32.768]
        self.optimal_x = [0, 0]  # Location
        self.resolution = 250
        self.z = None  # Function evaluation
        self.set_observation()
        self.set_domain()
        self.set_range()
        self.init_plot(env_params["data path"])

    def get_func(self):
        """Evaluate the function based on the position attribute."""
        a = 20
        b = 0.2
        c = 2*math.pi
        d = ((self.x[0]**2)+self.x[1]**2)*0.5
        e = -b*torch.sqrt(d)
        f = -a*torch.exp(e)
        g = torch.cos(torch.mul(self.x[0], c))+torch.cos(torch.mul(self.x[1], c))
        h = torch.exp(0.5*g)
        i = f-h+a+math.exp(1)
        return i









        #
