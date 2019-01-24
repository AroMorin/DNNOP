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
from .analysis import Analysis
from .perturbation import Perturbation

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
        self.perturb = Perturbation(hyper_params)
        self.analyzer = Analysis(hyper_params)
        self.scores = []
        self.get_weights_dicts()
        self.set_param_vecs()

    def set_new_pool(self, scores):
        self.analyzer.analyze(scores)

        self.elite.set_elite(self.param_vecs, self.analyzer)
        elite = self.elite.model

        self.anchors.set_anchors(self.param_vecs, self.analyzer, elite)
        anchors = self.anchors.models

        self.probes.set_probes(anchors, self.analyzer)
        probes = self.probes.models

        self.blends.set_blends(self.anchors.models, self.models, self.analyzer)
        blends = self.blends.models

        self.construct_pool()
        self.set_weight_dicts()
        self.update_models()

    def get_weights_dicts(self):
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

    def construct_pool(self):
        # Define noise magnitude and scale
        self.apply_perturbation(self.probes)
        self.apply_perturbation(self.blends)
        self.pool = []
        self.append_to_list(self.pool, self.anchors)
        self.append_to_list(self.pool, self.probes)
        self.append_to_list(self.pool, self.blends)
        assert len(self.pool) == self.hp.pool_size  # Sanity check

    def apply_perturbation(self, tensors):
        for t in tensors:
            self.perturb.apply(t, self.analyzer)

    def append_to_list(self, mylist, incoming):
        for item in incoming:
            mylist.append(item)
        return mylist

    def set_weight_dicts(self):
        """This method takes in parameter vectors and shapes them into weight
        dictionaries.
        """
        pass

    def update_models(self):
        """This function updates the ".parameters" of the models using the
        newly-constructed weight dictionaries.
        """
        pass






#
