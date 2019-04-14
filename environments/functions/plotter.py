"""Class for plotting functions."""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm, colors
import torch
import os

class Plotter:
    def __init__(self, func, data_path):
        self.elite_x = 0
        self.elite_y = 0
        self.elite_score = 0

        self.anchors_x = []
        self.anchors_y = []
        self.anchors_scores = []

        self.probes_x = []
        self.probes_y = []
        self.probes_scores = []

        self.probe_x = 0
        self.probe_y = 0
        self.probe_score = 0

        self.blends_x = []
        self.blends_y = []
        self.blends_scores = []

        self.integrity = 0
        self.iteration = 0
        self.backtracking = False
        self.data_path = data_path
        self.make_dir(data_path)
        self.top = None
        self.front = None
        self.artists = []
        self.net = []
        self.fig = None
        self.gs = None
        self.x1 = func.domain[0]
        self.x2 = func.domain[1]
        self.z = func.range.cpu().numpy()
        self.x1_low = 0
        self.x2_low = 0
        self.x1_high = 0
        self.x2_high = 0
        self.z_low = 0
        self.z_high = 0
        self.set_limits(func)
        self.z_levels = np.linspace(self.z_low, self.z_high, func.resolution)
        self.plot_base()

    def make_dir(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def set_limits(self, func):
        """Sets the limits of the plot in X1, X2 and Z axes."""
        self.x1_low = func.x_low[0]
        self.x2_low = func.x_low[1]
        self.x1_high = func.x_high[0]
        self.x2_high = func.x_high[1]
        if func.minimize:
            self.z_low = func.target
            self.z_high = np.max(self.z)
        else:
            self.z_low = np.min(self.z)
            self.z_high = func.target

    def plot_base(self):
        """Plots the function over the entire evaluation domain. This is what
        we call the "base". This base remains constant. We only add predictions
        (ie. artists) to it later.
        This method speeds up the plotting process, because we don't need to
        re-construct the base every iteration/generation.
        """
        self.init_fig()
        self.plot_top()
        self.plot_front()
        self.plot_iso()
        self.plot_colorbar()
        plt.tight_layout()

    def init_fig(self):
        """Initializes the plot figure to be of certain properties, e.g. size."""
        self.fig = plt.figure(figsize=(30, 10))
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3],
                                    figure=self.fig)
        self.cmap = cm.get_cmap('Spectral', len(self.z_levels))

    def plot_top(self):
        """Plots top view of the base."""
        self.top = plt.subplot(self.gs[0])
        self.CS = self.top.contourf(self.x1, self.x2, self.z,
                                self.z_levels,
                                cmap=self.cmap)
        self.top.set_title('Top View')
        self.top.set_xlabel('x1')
        self.top.set_ylabel('x2')
        self.top.set_xlim([self.x1_low, self.x1_high])
        self.top.set_ylim([self.x2_low, self.x2_high])

    def plot_front(self):
        """Plots front view of the base."""
        self.front = plt.subplot(self.gs[1])
        self.front.pcolormesh(self.x1, self.z, self.z,
                            cmap=self.cmap)
        self.front.set_title('Front View')
        self.front.set_xlabel('x1')
        self.front.set_ylabel('z')
        self.front.set_xlim([self.x1_low, self.x1_high])
        self.front.set_ylim([self.z_low, self.z_high])

    def plot_iso(self):
        """Plots isometric/3D view of the base."""
        self.iso = plt.subplot(self.gs[2], projection='3d')
        self.iso.plot_surface(self.x1, self.x2, self.z,
                            cmap=self.cmap)
        self.iso.set_title("3D view")
        self.iso.set_xlim([self.x1_low, self.x1_high])
        self.iso.set_ylim([self.x2_low, self.x2_high])
        self.iso.set_zlim([self.z_low, self.z_high])
        self.iso.set_xlabel('x1')
        self.iso.set_ylabel('x2')
        self.iso.set_zlabel('z')

    def plot_colorbar(self):
        """Adds a color bar to the base figure."""
        norm = colors.Normalize(vmin=self.z_low, vmax=self.z_high)
        mtop = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        mtop.set_array([])
        colorbar = self.fig.colorbar(mtop, orientation='vertical')
        colorbar.set_label('z')
        #colorbar = self.fig.colorbar(self.CS, cax=self.cb)

    def plot_net(self, positions, scores):
        """Plots the initial guesses of the networks."""
        x = [i[0].item() for i in positions]
        y = [i[1].item() for i in positions]
        scores = [i.item() for i in scores]
        marker = '^'
        s = 100
        color = 'black'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label='initials')
        d = self.fig.text(0.85, 0.12, 'Iteration: 0')
        self.net = [a, b, c, d]
        self.iso.legend(loc='best')
        self.save_figure()
        self.remove_artists()

    def plot_artists(self, alg, iteration):
        """Adds the guesses from the networks, saves the plot figure, then
        removes the artists.
        """
        self.update_state(alg, iteration)
        self.plot_anchors()
        self.plot_probes()
        self.plot_blends()
        self.plot_elite()
        self.iso.legend(loc='best')
        self.plot_text()
        self.save_figure()
        self.remove_artists()

    def update_state(self, alg, iteration):
        """Update plotter state."""
        self.backtracking = alg.pool.analyzer.backtracking
        self.integrity = alg.pool.analyzer.integrity
        self.iteration = iteration
        self.artists = []  # Reset state
        self.net = []  # Reset state

        elite = alg.inferences[alg.pool.elite.elite_idx]
        elite_score = alg.pool.elite.elite_score
        self.elite_x = elite[0].item()
        self.elite_y = elite[1].item()
        self.elite_score = elite_score.item()

        a = alg.pool.anchors.nb_anchors
        anchors = alg.inferences[1:a+1]
        anchors_scores = alg.optim.scores[1:a+1]
        assert len(anchors) == a  # Sanity check
        self.anchors_x = [i[0].item() for i in anchors]
        self.anchors_y = [i[1].item() for i in anchors]
        self.anchors_scores = [i.item() for i in anchors_scores]

        b = a+(alg.pool.anchors.nb_anchors*alg.optim.hp.nb_probes)
        probes = alg.inferences[a+1:b+1]
        probes_scores = alg.optim.scores[a+1:b+1]
        assert len(probes) == len(alg.pool.probes.probes_idxs)  # Sanity check
        self.probes_x = [i[0].item() for i in probes]
        self.probes_y = [i[1].item() for i in probes]
        self.probes_scores = [i.item() for i in probes_scores]

        blends = alg.inferences[b+1:]
        blends_scores = alg.optim.scores[b+1:]
        assert len(blends) == len(alg.pool.blends.blends_idxs)  # Sanity check
        self.blends_x = [i[0].item() for i in blends]
        self.blends_y = [i[1].item() for i in blends]
        self.blends_scores = [i.item() for i in blends_scores]

    def plot_artist(self, alg, iteration):
        self.update_state_single(alg, iteration)
        if alg.pool.elite.replaced_elite:
            self.plot_elite()
        else:
            self.plot_probe()
        self.plot_text()
        self.save_figure()
        self.remove_text()

    def update_state_single(self, alg, iteration):
        """Update plotter state."""
        self.backtracking = alg.pool.analyzer.backtracking
        self.integrity = alg.pool.analyzer.integrity
        self.iteration = iteration
        self.artists = []  # Reset state
        self.net = []  # Reset state

        elite = alg.pool.elite.inference
        elite_score = alg.pool.elite.elite_score
        self.elite_x = elite[0].item()
        self.elite_y = elite[1].item()
        self.elite_score = elite_score.item()

        self.probe_x = alg.inference[0].item()
        self.probe_y = alg.inference[1].item()
        self.probe_score = alg.optim.score.item()

    def plot_elite(self):
        """Plots the elite."""
        x = self.elite_x
        y = self.elite_y
        score = self.elite_score
        s = 100
        marker = '*'
        color = 'red'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, score, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, score, marker=marker, s=s, c=color, label = 'elite')
        self.artists.extend([a, b, c])

    def plot_probe(self):
        x = self.probe_x
        y = self.probe_y
        score = self.probe_score
        s = 100
        marker = '^'
        color = 'black'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, score, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, score, marker=marker, s=s, c=color, label = 'elite')
        self.artists.extend([a, b, c])

    def plot_anchors(self):
        """Plots the anchors."""
        x = self.anchors_x
        y = self.anchors_y
        scores = self.anchors_scores
        s = 100
        marker = 'x'
        color = 'blue'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'anchors')
        self.artists.extend([a, b, c])

    def plot_probes(self):
        """Plots probes."""
        x = self.probes_x
        y = self.probes_y
        scores = self.probes_scores
        s = 100
        marker = '.'
        color = 'green'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'probes')
        self.artists.extend([a, b, c])

    def plot_blends(self):
        """Plots blends."""
        x = self.blends_x
        y = self.blends_y
        scores = self.blends_scores
        s = 100
        marker = '+'
        color = 'yellow'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'blends')
        self.artists.extend([a, b, c])

    def plot_text(self):
        """Adds text to the plot figure. Text conveys important information
        such as current elte, value of integrity, etc...
        """
        a = self.fig.text(0.85, 0.12, 'Iteration: '+str(self.iteration))
        score_str = 'Elite Score: '+str("{0:.4g}".format(self.elite_score))
        b = self.fig.text(0.85, 0.1, score_str)
        pos_str = 'Elite Position: ('+str(
                    "{0:.4g}".format(self.elite_x))+', '+str(
                    "{0:.4g}".format(self.elite_y))+')'
        c = self.fig.text(0.85, 0.08, pos_str)
        integrity_str = 'Integrity: '+str("{0:.4g}".format(self.integrity))
        d = self.fig.text(0.85, 0.06, integrity_str)
        e = self.fig.text(0.85, 0.04, '')
        if self.backtracking:
            e = self.fig.text(0.85, 0.04, 'Backtracking activated!')
        self.artists.extend([a, b, c, d, e])

    def save_figure(self):
        """Saves the plot as a figue on disk/storage."""
        fn = self.data_path+str(self.iteration)+'.png'
        self.fig.savefig(fn)

    def remove_artists(self):
        """Removes the plot artists from the figure."""
        if len(self.net)==0:
            assert len(self.artists) == 17
            for artist in self.artists:
                artist.remove()
        else:
            for plot in self.net:
                plot.remove()

    def remove_text(self):
        """Removes the plot artists from the figure."""
        if len(self.net)==0:
            for artist in self.artists[-5:]:
                artist.remove()
        else:
            for plot in self.net:
                plot.remove()
