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

import torch

class Pool:
    def __init__(self, models, hyper_params):
        self.models = models # List of Models
        self.state_dicts = [] # List of weight dictionaries
        self.param_vecs = [] # List of parameter vectors
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.elite_dict = {}
        self.hp = hyper_params
        self.elite = Elite(hyper_params)
        self.anchors = Anchors(hyper_params)
        self.probes = Probes(hyper_params)
        self.blends = Blends(hyper_params)
        self.perturb = Perturbation(hyper_params)
        self.analyzer = Analysis(hyper_params)
        self.scores = []
        self.set_state_dicts()
        self.set_param_vecs()

    def prep_new_pool(self, scores):
        self.analyzer.analyze(scores)

        self.elite.set_elite(self.param_vecs, self.analyzer)
        elite = self.elite.model

        self.anchors.set_anchors(self.param_vecs, self.analyzer, elite)
        anchors = self.anchors.models

        exit()

        self.probes.set_probes(anchors, self.analyzer)
        probes = self.probes.models

        self.blends.set_blends(self.anchors.models, self.models, self.analyzer)
        blends = self.blends.models

        self.construct_pool()
        self.set_weight_dicts()
        self.update_models()

    def set_state_dicts(self):
        """This method takes in the list of models, i.e. pool, and produces
        a list of weight dictionaries.
        """
        for model in self.models:
            self.state_dicts.append(model.state_dict())
        self.nb_layers = len(model.state_dict())

    def set_param_vecs(self):
        """This method takes in the list of weight dictionaries and produces
        a list of parameter vectors.
        Note: parameter vectors are essentially "flattened" weights.
        """
        for state_dict in self.state_dicts:
            vec = self.dict_to_vec(state_dict)
            self.param_vecs.append(vec)

    def dict_to_vec(self, dict):
        vec = torch.empty(self.nb_layers)
        for i, key in enumerate(dict):
            x = torch.tensor(dict[key])
            self.shapes.append(x.size())
            self.num_elems.append(x.numel())
            print(x.size())
            print(x.numel())
            vec[i] = x.reshape(1, 250)
        print (mylist)
        print (len(mylist))
        print(type(mylist[0]))
        print(mylist[0].size())
        print(mylist[1].size())
        vec = torch.as_tensor(mylist)
        print (vec.size())
        print (type(vec))
        print (type(vec[0]))
        print (vec[0].size())
        exit()
        return vec

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
