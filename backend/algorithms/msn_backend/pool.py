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
    def __init__(self, models, hyper_params):
        self.models = models # List of Models
        self.weights_dicts = [] # List of weight dictionaries
        self.param_vecs = [] # List of parameter vectors
        self.hp = hyper_params
        self.elite = Elite(hyper_params)
        self.anchors = Anchors(hyper_params)
        self.probes = Probes(hyper_params)
        self.blends = Blends(hyper_params)
        self.scores = []
        self.set_weights_dicts()
        self.set_param_vecs()

    def set_weights_dicts(self):
        """This method takes in the list of models, i.e. pool, and produces
        a list of weight dictionaries.
        """
        #self.weights_dicts =
        pass

    def set_param_vecs(self):
        """This method takes in the list of weight dictionaries and produces
        a list of parameter vectors.
        Note: parameter vectors are essentially "flattened" weights.
        """
        #self.param_vecs =
        pass

    def set_new_pool(self, scores):
