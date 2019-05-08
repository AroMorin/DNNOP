"""Base class for novelty score"""


class Novelty(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.table = []
        self.counts = []
        self.count = 0
        self.limit = 0.1
        self.factor = 1.02
        self.idx = None
        self.forget_counter = 3

    def update(self, item):
        item = item.item()
        if self.in_table(item):
            self.set_penalty(item)
            self.append_count(item)
        else:
            self.value = 0.
            self.append_table(item)

    def set(self, item):
        if self.in_table(item):
            self.set_elite_penalty(item)
            self.append_count(item)
        else:
            self.value = 0.
            self.append_table(item)
        self.forget()


    def in_table(self, item):
        if item in self.table:
            self.idx = self.table.index(item)
            self.count = self.counts[self.idx]  # Update count of item
            return True
        else:
            return False

    def set_penalty(self, item):
        if self.hp.minimizing:
            print(self.count)
            self.value = item*((self.factor**self.count)-1)
            if self.value < 0.:
                self.value = -self.value
            print("penalty: %f" %self.value)
        else:
            self.value = -item*((self.factor**self.count)-1)

    def set_elite_penalty(self, item):
        if self.hp.minimizing:
            print(self.count)
            self.value = item*((self.factor**(self.count-1))-1)
            if self.value < 0.:
                self.value = -self.value
            print("penalty: %f" %self.value)
        else:
            self.value = -item*((self.factor**(self.count-1))-1)

    def set_reward(self, item):
        if self.hp.minimizing:
            self.value = -(1/item)*self.factor
        else:
            self.value = item*self.factor
        self.value = 0.

    def append_table(self, item):
        self.table.append(item)
        self.counts.append(1)

    def append_count(self, item):
        self.counts[self.idx]+=1

    def forget(self):
        if self.forget_counter == 0:
            self.reduce_count()
            self.forget_counter = 3
        else:
            self.forget_counter -=1

    def reduce_count(self):
        self.counts = [i-1 for i in self.counts]
        while 0 in self.counts:
            self.idx = self.counts.index(0)
            del self.table[self.idx]
            del self.counts[self.idx]
        assert 0 not in self.counts
        assert len(self.counts) == len(self.table)

#
