"""Base class for novelty score"""

from scipy import interpolate
import numpy as np
from collections import deque, Counter

class Frustration(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.mem_size = 50
        self.table = deque([], maxlen=self.mem_size)
        self.counts = Counter()
        self.penalties = {}
        self.factor = 1.15
        self.sim = None
        self.min_len = 4
        self.uniques = 0
        self.epsilon = 0.001
        self.patience = 10

    def update(self, item):
        item = item.item()
        self.append_table(item)
        self.uniques = len(set(self.table))
        if self.uniques<=self.min_len:
            self.construct_hybrids()
        self.update_count()
        self.update_penalties()
        print(self.table)
        print(self.counts)
        self.update_sim()

    def append_table(self, item):
        self.table.append(item)

    def update_count(self):
        self.counts = Counter(self.table)

    def update_penalties(self):
        self.penalties = {}
        for element in set(self.table):
            if self.hp.minimizing:
                pen = element*((self.factor**self.counts[element])-1)
                if pen < 0.:
                    print(pen, element)
                    exit()
                    pen = -pen
            else:
                pen = -element*((self.factor**self.counts[element])-1)
                if pen > 0.:
                    pen = -pen
            if pen<0.:
                print(pen, element)
                exit()
            self.penalties[element] = pen

    def construct_hybrids(self):
        hybrids = []
        i=0
        while self.uniques < self.min_len:
            h = self.table[i]+(np.random.randint(1, 5, size=1)*self.epsilon)
            self.table.append(h[0])
            self.uniques = len(set(self.table))
            i +=1
            if i==len(self.table):
                i = 0

    def update_sim(self):
        x = np.array(list(self.penalties.keys()))
        y = np.array(list(self.penalties.values()))
        self.sim = interpolate.interp1d(x, y, bounds_error=False, fill_value=0.,
                                        kind='linear')

    def set_jump_p(self):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = 1.-integrity
        argument = (self.hp.lambda_*p)-2.5
        exp1 = math.tanh(argument)+1
        self.search_radius = exp1*self.hp.lr

    def set_penalty(self, item):
        item = item.item()
        if pick == 1:
            self.value = 0.
        else:
            value = self.sim(item)
            self.value = value.tolist()
        print("penalty: %f" %self.value)

#
