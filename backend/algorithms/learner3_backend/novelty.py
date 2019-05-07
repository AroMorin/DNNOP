"""Base class for novelty score"""


class Novelty(object):
    def __init__(self, hp):
        self.hp = hp
        self.value = 0.
        self.table = []
        self.limit = 0.1
        self.factor = 0.01
        self.count = 0

    def update(self, item):
        if self.in_table(item):
            self.set_penalty(item)
        else:
            self.set_reward(item)

    def set(self, item):
        if self.in_table(item):
            self.set_elite_penalty(item)
        else:
            self.set_reward(item)

    def in_table(self, item):
        if item in self.table:
            self.count = self.table.count(item)  # Update count of item
            print(self.count)
            return True
        else:
            return False

    def set_penalty(self, item):
        if self.hp.minimizing:
            self.value = item*self.factor*(self.count)
        else:
            self.value = -item*self.factor*(self.count)

    def set_elite_penalty(self, item):
        if self.hp.minimizing:
            self.value = item*self.factor*(self.count-1)
        else:
            self.value = -item*self.factor*(self.count-1)

    def set_reward(self, item):
        if self.hp.minimizing:
            self.value = -item*self.factor
        else:
            self.value = item*self.factor

    def append_table(self, item):
        item = item.item()
        self.table.append(item)




#
