"""Class for plotting functions"""

class Plotter:
    def __init__(self):
        self.top = None
        self.artists = None
        self.fig = None
        self.x_1 = 0
        self.x_2 = 0
        self.y_1 = 0
        self.y_2 = 0
        self.domain = 0
        # Range of the function, not range of axes
        self.range = range
        self.color_scheme = None
        self.step = 0

    def set_domain(self):
        pass

    def plot_base(self):
        pass

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
