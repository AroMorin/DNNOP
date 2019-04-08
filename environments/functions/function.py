"""Base class for functions."""

import numpy as np
import torch
from .plotter import Plotter
from ..environment import Environment

class Function(Environment):
    def __init__(self, env_params):
        super(Function, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.optimal_x = 0  # Location
        self.resolution = 50
        self.x_low = [0, 0]
        self.x_high = [0, 0]
        self.domain = []  # Matrix of coordinate vectors
        self.range = []  # Matrix of function values
        self.iteration = -1  # State
        self.plot = env_params["plot"]

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "data path": "C:/",
                            "plot": False,
                            "precision": torch.float,
                            "scoring type": "error"
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def init_plot(self, data_path):
        """Instantiates the plotter class if a plot is requested by the user."""
        if self.plot:
            self.plotter = Plotter(self, data_path)

    def set_observation(self):
        """Randomly picks a point within the function domain as an origin for
        the search process.
        """
        origin_x1 = np.random.uniform(self.x_low[0], self.x_high[0], 1)
        origin_x2 = np.random.uniform(self.x_low[1], self.x_high[1], 1)
        origin = [origin_x1[0], origin_x2[0]]
        print("Origin: ", origin)
        self.observation = [torch.tensor(origin,
                                        dtype=self.precision,
                                        device=self.device),
                            self.x_low,
                            self.x_high]

    def set_domain(self):
        """Sets the meshgrid domain for the function."""
        x1 = torch.linspace(self.x_low[0], self.x_high[0], self.resolution)
        x2 = torch.linspace(self.x_low[1], self.x_high[1], self.resolution)
        m1, m2 = torch.meshgrid(x1, x2)
        self.domain = [m1, m2]

    def set_range(self):
        """Sets the function range, given the domain."""
        self.x = self.domain
        self.range = self.get_func()

    def evaluate(self, position):
        """Evaluates the function given an (x1, x2) coordinate."""
        #x1 = position[0].cpu().numpy()
        #x2 = position[1].cpu().numpy()
        #self.x = [x1, x2]
        self.x = position
        self.z = self.get_func()
        return self.z

    def make_plot(self, alg):
        """Plots the algorithm's predictions on canvas, and saves the plot as
        a figure on disk/storage.
        """
        if self.iteration != 0:
            self.plotter.plot_artists(alg, self.iteration)
        else:
            positions = alg.inferences
            scores = alg.optim.scores
            self.plotter.plot_net(positions, scores)

    def step(self):
        """Steps the environment."""
        self.iteration += 1

#
