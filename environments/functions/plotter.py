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
        self.x1_low = self.x1[0]
        self.x2_low = self.x2[0]
        self.x1_high = self.x1[-1]
        self.x2_high = self.x2[-1]
        self.z_low = np.min(self.z)
        self.z_high = np.max(self.z)
        self.N = np.linspace(self.z_low, self.z_high, func.resolution)
        self.plot_base()

    def plot_base(self):
        self.init_fig()
        self.plot_top()
        self.plot_front()
        self.plot_iso()
        plt.tight_layout()
        plt.show()
        exit()

    def init_fig(self):
        self.fig = plt.figure(figsize=(30, 10))
        self.gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])

    def plot_top(self):
        self.top = plt.subplot(self.gs[0])
        CS = self.top.contourf(self.x1, self.x2, self.z, self.N, cmap='Spectral')
        self.top.set_title('Top View')
        self.top.set_xlabel('x1')
        self.top.set_ylabel('x2')
        cbar = self.fig.colorbar(CS, ax=self.top, ticks=[self.z_low, self.z_high])
        cbar.ax.set_ylabel('z')

    def plot_front(self):
        self.front = plt.subplot(self.gs[1])
        self.front.contour(self.x1, self.z, self.x2, cmap='Spectral')
        self.front.set_title('Front View')
        self.front.set_xlabel('x1')
        self.front.set_ylabel('z')

    def plot_iso(self):
        self.iso = plt.subplot(self.gs[2], projection='3d')
        self.iso.plot_surface(self.x1, self.x2, self.z, cmap='Spectral')

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
