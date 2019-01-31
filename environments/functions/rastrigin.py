"""Implementation of the Rastrigin function as in the link below. The number of
problem dimensions is arbitrary, as well as the bounds.
https://www.sfu.ca/~ssurjano/rastr.html
"""
from .function import Function
from .plotter import Plotter

class Rastrigin(Function):
    def __init__(self, nb_dimensions, plot):
        self.nb_dimensions = nb_dimensions
        self.plot = plot
        self.x = None  # NP array
        self.x1 = -5.12
        self.x2 = 5.12
        self.y1 = -5.12
        self.y2 = 5.12
        self.z = 0
        self.range = set_range()
        self.init_plot()

    def init_plot(self):
        if self.plot:
            self.plotter = Plotter()

    def get_func(self):
        a = 10*self.nb_dimensions
        b = np.square(self.x)
        c = 10*np.cosine(2*np.pi*self.x)
        d = b - c
        e = np.sum(d)
        return a + e

    def evaluate(self, x):
        self.x = x
        self.z = self.get_func()

    def step(self):
        pass










        #
