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
        self.factor = 1.5
        self.increment = 1.
        self.decrement = 0.5
        self.f = 1
        self.forget_counter = self.f
        self.sim = None
        self.min_len = 3
        self.mem_len = 5

    def update(self, item):
        item = item.item()
        if item in self.table:
            self.increment_count(item)
        else:
            self.append_lists(item)
        self.forget()
        self.update_penalties()
        if len(self.table)>=self.min_len:
            self.update_sim()
        print(self.table)
        print(self.counts)

    def set_penalty(self, item):
        item = item.item()
        if len(self.table)<self.min_len:
            self.value = 0.
        else:
            value = self.sim(item)
            self.value = value.tolist()
        print("penalty: %f" %self.value)

    def append_lists(self, item):
        self.table.append(item)
        self.counts.append(self.increment)
        self.penalties.append(0.)

    def increment_count(self, item):
        idx = self.table.index(item)
        self.counts[idx]+=self.increment

    def forget(self):
        if self.forget_counter == 0:
            self.reduce_count()
            self.forget_counter = self.f
        else:
            self.forget_counter -=1

    def reduce_count(self):
        self.counts = [max(0, i-self.decrement) for i in self.counts]
        while 0 in self.counts and len(self.table)>self.min_len :
            idx = self.counts.index(0)
            del self.table[idx]
            del self.counts[idx]
            del self.penalties[idx]
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
        #self.sim = interpolate.interp1d(x, y, fill_value=0., kind='linear')
        z = 0.
        self.sim = interpolate.interp1d(x, y, bounds_error=False,
                                        fill_value=z, kind='linear')


#
