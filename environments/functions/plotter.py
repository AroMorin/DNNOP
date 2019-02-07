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

    def plot_base(self, x, y, levels):
        print(x.shape)
        print(len(x[1]))
        plt.figure()
        plt.contourf(x[0], x[1], y)
        plt.show()
        exit()

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
