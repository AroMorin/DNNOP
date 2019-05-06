"""Base class for novelty score"""

import torch
import numpy as np

class Novelty(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.table = []
        self.limit = 0.1
        self.factor = 0.01
        self.count = 0

    def update(self, item):
        item = item.item()
        if self.in_table(item):
            self.set_penalty(item)
        else:
            self.set_score(item)
            self.append_table(item)

    def set_score(self, item):
        if self.hp.minimizing:
            self.value = -item*self.factor
        else:
            self.value = item*self.factor

    def set_penalty(self, item):
        if self.hp.minimizing:
            self.value = item*self.factor*self.count
        else:
            self.value = -item*self.factor*self.count

    def in_table(self, item):
        if item in self.table:
            self.count = self.table.count(item)
            return True
        else:
            return False

    def append_table(self, item):
        self.table.append(item)

#
