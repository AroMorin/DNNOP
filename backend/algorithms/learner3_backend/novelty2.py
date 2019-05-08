"""Base class for novelty score"""

from scipy import interpolate
import numpy as np

class Novelty(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.table = []
        self.counts = []
        self.penalties = []
        self.limit = 0.1
        self.factor = 1.02
        self.idx = None
        self.f = 2
        self.forget_counter = self.f
        self.sim = None

    def update(self, item):
        item = item.item()
        if item in self.table:
            self.increment_count(item)
        else:
            self.append_lists(item)
        self.forget()
        self.update_penalties()
        if len(self.table)>3:
            self.update_sim()
        print(self.counts)
        print((self.table))
        print(self.penalties)

    def set_penalty(self, item):
        if len(self.table)<4:
            self.value = 0.
        else:
            value = self.sim(item.item())
            self.value = value.tolist()
        print("penalty: %f" %self.value)

    def append_lists(self, item):
        self.table.append(item)
        self.counts.append(1)
        self.penalties.append(0.)

    def increment_count(self, item):
        idx = self.table.index(item)
        self.counts[idx]+=1

    def forget(self):
        if self.forget_counter == 0:
            self.reduce_count()
            self.forget_counter = self.f
        else:
            self.forget_counter -=1

    def reduce_count(self):
        self.counts = [i-1 for i in self.counts]
        while 0 in self.counts:
            idx = self.counts.index(0)
            self.counts[idx]+=1
        assert 0 not in self.counts
        assert len(self.counts) == len(self.table)

    def update_penalties(self):
        assert len(self.penalties) == len(self.table)
        assert len(self.penalties) == len(self.counts)
        for idx, element in enumerate(self.table):
            if self.hp.minimizing:
                pen = element*((self.factor**self.counts[idx])-1)
                if pen < 0.:
                    pen = -pen
            else:
                pen = -element*((self.factor**self.counts[idx])-1)
                if pen > 0.:
                    pen = -pen
            self.penalties[idx] = pen

    def update_sim(self):
        x = np.array(self.table)
        y = np.array(self.penalties)
        idxs = np.argsort(x)
        x = x[idxs]
        y = y[idxs]
        print (x)
        print(y)
        #self.sim = interpolate.interp1d(x, y, fill_value=0., kind='linear')
        self.sim = interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')


#
