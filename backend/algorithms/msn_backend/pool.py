"""base class for pool
Its functions are:
1) Initializing the pool with the given random conditions
2) Returning/Setting the pool
3) Sorting the pool members according to their performance
4) Maintain the pool composition of Elite, Anchors and Probes


The pool object will contain the models under optimization.
"""

class Pool():
    def __init__(self, nb_samples, models):
        self.nb_samples = nb_samples
        self.models = ''
        self.weight_dicts = '' # Weight dictionaries
        self.pool = '' # Parameter vectors
        self.init_models(self)

    def init_models(self):
