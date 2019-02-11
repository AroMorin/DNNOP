"""Class for plotting functions"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm, colors

class Plotter:
    def __init__(self, func):
        self.top = None
        self.front = None
        self.artists = None
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
        plt.show()
        exit()

    def init_fig(self):
        self.fig = plt.figure(figsize=(30, 10))
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3],
                                    figure=self.fig)
        self.cmap = cm.get_cmap('Spectral', len(self.z_levels))

    def plot_top(self):
        self.top = plt.subplot(self.gs[0])
        self.CS = self.top.contourf(self.x1, self.x2, self.z,
                                self.z_levels,
                                vmin=self.z_low, vmax=self.z_high,
                                cmap=self.cmap)
        self.top.set_title('Top View')
        self.top.set_xlabel('x1')
        self.top.set_ylabel('x2')

    def plot_front(self):
        self.front = plt.subplot(self.gs[1])
        self.front.pcolormesh(self.x1, self.z, self.z,
                            cmap=self.cmap)
        self.front.set_title('Front View')
        self.front.set_xlabel('x1')
        self.front.set_ylabel('z')

    def plot_iso(self):
        self.iso = plt.subplot(self.gs[2], projection='3d')
        self.iso.plot_surface(self.x1, self.x2, self.z,
                            vmin=self.z_low, vmax=self.z_high,
                            cmap=self.cmap)

    def plot_colorbar(self):
        norm = colors.Normalize(vmin=self.z_low, vmax=self.z_high)
        mtop = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        mtop.set_array([])
        colorbar = self.fig.colorbar(mtop, orientation='vertical')
        colorbar.set_label('z')
        #colorbar = self.fig.colorbar(self.CS, cax=self.cb)

    def plot_anchors(self, anchors):
        pass

    def plot_probes(self, probes):
        pass

    def plot_blends(self, blends):
        pass

    def plot_elite(self, elite):
        pass

    def save_figure(self):
        pass
