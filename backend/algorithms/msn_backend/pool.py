"""base class for pool
Its functions are:
1) Initializing the pool with the given random conditions
2) Returning/Setting the pool
3) Sorting the pool members according to their performance
4) Maintain the pool composition of Elite, Anchors and Probes


The pool object will contain the models under optimization.
"""
from .anchors import Anchors
from .probes import Probes
from .blends import Blends
from .elite import Elite

class Pool:
    def __init__(self, pool, hyper_params):
        # List of Models
        self.pool = pool
        # List of weight dictionaries
        self.weight_dicts = ''
        # List of Parameter Vectors
        self.parameters = ''
        self.anchors = Anchors(hyper_params)
        self.probes = Probes(hyper_params)
        self.blends = Blends(hyper_params)
        self.elite = Elite(hyper_params)
        # List of scores/fitness for every sample in the pool
        self.scores = ''

    def set_param_vector(self):
        pass

    def set_weight_dicts(self):
        """This function uses the values in the parameters variable to
        update the current weight dictionaries.
        It is not clear to me whether this will be followed by an update to
        the pool. I don't think so, since the pool is pointing to the model
        objects, not their weights.
        """
