"""Base class for functions"""

import numpy as np
import torch
from .plotter import Plotter
from ..environment import Environment

class Function(Environment):
    def __init__(self, plot, precision):
        super().__init__(precision)
        self.plot = plot
        self.optimal_x = 0  # Location
        self.resolution = 50
        self.x_low = [0, 0]
        self.x_high = [0, 0]
        self.domain = []  # Matrix of coordinate vectors
        self.range = []  # Matrix of function values
        self.score = True
        self.iteration = -1  # State

    def init_plot(self, data_path):
        if self.plot:
            self.plotter = Plotter(self, data_path)

    def set_observation(self):
        origin_x1 = np.random.uniform(self.x_low[0], self.x_high[0], 1)
        origin_x2 = np.random.uniform(self.x_low[1], self.x_high[1], 1)
        origin = [origin_x1[0], origin_x2[0]]
        print("Origin: ", origin)
        self.observation = [torch.tensor(
                            origin,
                            dtype=self.precision,
                            device = self.device),
                            self.x_low,
                            self.x_high]

    def set_domain(self):
        x1 = np.linspace(self.x_low[0], self.x_high[0], self.resolution)
        x2 = np.linspace(self.x_low[1], self.x_high[1], self.resolution)
        m1, m2 = np.meshgrid(x1, x2)
        self.domain = [m1, m2]

    def set_range(self):
        self.x = self.domain
        self.range = self.get_func()

    def construct_base(self):
        pass

    def evaluate(self, position):
        x1 = position[0].cpu().numpy()
        x2 = position[1].cpu().numpy()
        self.x = [x1, x2]
        self.z = self.get_func()
        return self.z

    def make_plot(self, alg):
        if self.iteration != 0:
            positions, scores = self.get_artists(alg)
            self.plotter.plot_artists(positions, scores, self.iteration)
        else:
            positions = alg.inferences
            scores = alg.scores
            self.plotter.plot_net(positions, scores)

    def step(self):
        self.iteration += 1

    def get_artists(self, alg):
        elite = alg.inferences[0]
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
