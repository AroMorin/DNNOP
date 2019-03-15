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
        self.score = True
        self.iteration = -1  # State
        self.plot = env_params["plot"]
        self.init_plot(env_params["data path"])

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        if "data path" not in env_params:
            env_params["data path"] = "C:/"
        if "plot" not in env_params:
            env_params["plot"] = False
        if "precision" not in env_params:
            env_params["precision"] = torch.float
        return env_params

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
        x1 = np.linspace(self.x_low[0], self.x_high[0], self.resolution)
        x2 = np.linspace(self.x_low[1], self.x_high[1], self.resolution)
        m1, m2 = np.meshgrid(x1, x2)
        self.domain = [m1, m2]

    def set_range(self):
        """Sets the function range, given the domain."""
        self.x = self.domain
        self.range = self.get_func()

    def construct_base(self):
        """Not sure what this does, remove?"""
        pass

    def evaluate(self, position):
        """Evaluates the function given an (x1, x2) coordinate."""
        x1 = position[0].cpu().numpy()
        x2 = position[1].cpu().numpy()
        self.x = [x1, x2]
        self.z = self.get_func()
        return self.z

    def make_plot(self, alg):
        """Plots the algorithm's predictions on canvas, and saves the plot as
        a figure on disk/storage.
        """
        if self.iteration != 0:
            positions, scores = self.get_artists(alg)
            self.plotter.plot_artists(positions, scores, alg, self.iteration)
        else:
            positions = alg.inferences
            scores = alg.scores
            self.plotter.plot_net(positions, scores)

    def step(self):
        """Steps the environment."""
        self.iteration += 1

    def get_artists(self, alg):
        """Acquires the predictions and corresponding scores/evaluations from
        the algorithm object. Every category needs to be distinguished.
        """
        elite = alg.inferences[alg.optim.pool.elite.elite_idx]
        elite_score = alg.optim.pool.elite.elite_score
        a = alg.optim.pool.anchors.nb_anchors
        anchors = alg.inferences[1:a+1]
        anchors_scores = alg.scores[1:a+1]
        assert len(anchors) == a  # Sanity check
        b = a+(alg.optim.pool.anchors.nb_anchors*alg.optim.hp.nb_probes)
        probes = alg.inferences[a+1:b+1]
        probes_scores = alg.scores[a+1:b+1]
        assert len(probes) == len(alg.optim.pool.probes.probes_idxs)  # Sanity check
        blends = alg.inferences[b+1:]
        blends_scores = alg.scores[b+1:]
        assert len(blends) == len(alg.optim.pool.blends.blends_idxs)  # Sanity check

        positions = {
                    "elite": elite,
                    "anchors": anchors,
                    "probes":probes,
                    "blends":blends}
        scores = {
                    "elite": elite_score,
                    "anchors": anchors_scores,
                    "probes":probes_scores,
                    "blends":blends_scores}
        return positions, scores

#
