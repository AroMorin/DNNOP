"""Class for plotting functions"""

import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        self.top = None
        self.artists = None
        self.fig = None
        self.x_1 = 0
        self.x_2 = 0
        self.y_1 = 0
        self.y_2 = 0
        self.color_scheme = None
        self.step = 0

    def plot_base(self, x, z, N):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        grid = plt.GridSpec(1, 2, wspace=0.5)
        ax1 = plt.subplot(grid[0])
        ax2 = plt.subplot(grid[1])
        plt.show()
        exit()
        CS = ax1.contourf(x[0], x[1], z, N)
        ax1.set_title('Top View')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel('z')
        ax2.contourf(x[0], z, x[1], N)
        ax2.set_title('Front View')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('z')

    def init_fig(self):
        pass

    def plot_top(self):
        pass

    def plot_front(self):
        pass

    def plot_iso(self):
        pass

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
