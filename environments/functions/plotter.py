"""Class for plotting functions"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm, colors

class Plotter:
    def __init__(self, func, data_path):
        self.elite_score = 0
        self.elite_position = []
        self.integrity = 0
        self.iteration = 0
        self.backtracking = False
        self.data_path = data_path
        self.top = None
        self.front = None
        self.artists = []
        self.net = []
        self.fig = None
        self.gs = None
        self.x1 = func.domain[0]
        self.x2 = func.domain[1]
        self.z = func.range
        self.x1_low = 0
        self.x2_low = 0
        self.x1_high = 0
        self.x2_high = 0
        self.z_low = 0
        self.z_high = 0
        self.set_limits(func)
        self.z_levels = np.linspace(self.z_low, self.z_high, func.resolution)
        self.plot_base()

    def set_limits(self, func):
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
        self.init_fig()
        self.plot_top()
        self.plot_front()
        self.plot_iso()
        self.plot_colorbar()
        plt.tight_layout()

    def init_fig(self):
        self.fig = plt.figure(figsize=(30, 10))
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3],
                                    figure=self.fig)
        self.cmap = cm.get_cmap('Spectral', len(self.z_levels))

    def plot_top(self):
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
        self.front = plt.subplot(self.gs[1])
        self.front.pcolormesh(self.x1, self.z, self.z,
                            cmap=self.cmap)
        self.front.set_title('Front View')
        self.front.set_xlabel('x1')
        self.front.set_ylabel('z')
        self.front.set_xlim([self.x1_low, self.x1_high])
        self.front.set_ylim([self.z_low, self.z_high])

    def plot_iso(self):
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
        norm = colors.Normalize(vmin=self.z_low, vmax=self.z_high)
        mtop = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        mtop.set_array([])
        colorbar = self.fig.colorbar(mtop, orientation='vertical')
        colorbar.set_label('z')
        #colorbar = self.fig.colorbar(self.CS, cax=self.cb)

    def plot_net(self, positions, scores):
        x = [i[0].item() for i in positions]
        y = [i[1].item() for i in positions]
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

    def plot_artists(self, positions, scores, alg, iteration):
        self.update_state(alg, iteration)
        self.plot_anchors(positions["anchors"], scores["anchors"])
        self.plot_probes(positions["probes"], scores["probes"])
        self.plot_blends(positions["blends"], scores["blends"])
        self.plot_elite(positions["elite"], scores["elite"])
        self.iso.legend(loc='best')
        self.plot_text()
        self.save_figure()
        self.remove_artists()

    def update_state(self, alg, iteration):
        self.backtracking = alg.optim.pool.analyzer.backtracking
        self.integrity = alg.optim.pool.analyzer.integrity
        self.iteration = iteration
        self.elite_position = alg.inferences[0]
        self.elite_score = alg.optim.pool.elite.elite_score
        self.artists = []  # Reset state
        self.net = []  # Reset state

    def plot_text(self):
        a = self.fig.text(0.85, 0.12, 'Iteration: '+str(self.iteration))
        score_str = 'Elite Score: '+str("{0:.4g}".format(self.elite_score))
        b = self.fig.text(0.85, 0.1, score_str)
        pos_str = 'Elite Position: ('+str(
                    "{0:.4g}".format(self.elite_position[0]))+', '+str(
                    "{0:.4g}".format(self.elite_position[1]))+')'
        c = self.fig.text(0.85, 0.08, pos_str)
        integrity_str = 'Integrity: '+str("{0:.4g}".format(self.integrity))
        d = self.fig.text(0.85, 0.06, integrity_str)
        e = self.fig.text(0.85, 0.04, '')
        if self.backtracking:
            e = self.fig.text(0.85, 0.04, 'Backtracking activated!')
        self.artists.extend([a, b, c, d, e])

    def plot_elite(self, position, score):
        x = position[0].item()
        y = position[1].item()
        s = 100
        marker = '*'
        color = 'red'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, score, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, score, marker=marker, s=s, c=color, label = 'elite')
        self.artists.extend([a, b, c])

    def plot_anchors(self, positions, scores):
        x = [i[0].item() for i in positions]
        y = [i[1].item() for i in positions]
        s = 100
        marker = 'x'
        color = 'blue'
        a = self.top.scatter(x, y, marker=marker, s=s)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'anchors')
        self.artists.extend([a, b, c])

    def plot_probes(self, positions, scores):
        x = [i[0].item() for i in positions]
        y = [i[1].item() for i in positions]
        s = 100
        marker = '.'
        color = 'green'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'probes')
        self.artists.extend([a, b, c])

    def plot_blends(self, positions, scores):
        x = [i[0].item() for i in positions]
        y = [i[1].item() for i in positions]
        s = 100
        marker = '+'
        color = 'yellow'
        a = self.top.scatter(x, y, marker=marker, s=s, c=color)
        b = self.front.scatter(x, scores, marker=marker, s=s, c=color)
        c = self.iso.scatter(x, y, scores, marker=marker, s=s, c=color, label = 'blends')
        self.artists.extend([a, b, c])

    def save_figure(self):
        fn = self.data_path+str(self.iteration)+'.png'
        self.fig.savefig(fn)

    def remove_artists(self):
        if len(self.net)==0:
            assert len(self.artists) == 17
            for artist in self.artists:
                artist.remove()
        else:
            for plot in self.net:
                plot.remove()
