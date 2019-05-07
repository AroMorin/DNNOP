"""Base class for novelty score"""


class Novelty(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.table = []
        self.counts = []
        self.count = 0
        self.limit = 0.1
        self.factor = 0.01
        self.idx = None

    def update(self, item):
        item = item.item()
        if self.in_table(item):
            self.set_penalty(item)
            self.append_count(item)
        else:
            self.set_reward(item)
            self.append_table(item)

    def set(self, item):
        if self.in_table(item):
            self.set_elite_penalty(item)
        else:
            self.set_reward(item)

    def in_table(self, item):
        if item in self.table:
            self.idx = self.table.index(item)
            self.count = self.counts[self.idx]  # Update count of item
            print(self.count)
            return True
        else:
            return False

    def set_penalty(self, item):
        if self.hp.minimizing:
            self.value = (1/item)*self.factor*(self.count)
        else:
            self.value = -item*self.factor*(self.count)

    def set_elite_penalty(self, item):
        if self.hp.minimizing:
            self.value = (1/item)*self.factor*(self.count-1)
        else:
            self.value = -item*self.factor*(self.count-1)

    def set_reward(self, item):
        if self.hp.minimizing:
            self.value = -(1/item)*self.factor
        else:
            self.value = item*self.factor

    def append_table(self, item):
        self.table.append(item)
        self.counts.append(1)

    def append_count(self, item):
        self.counts[self.idx]+=1


#
