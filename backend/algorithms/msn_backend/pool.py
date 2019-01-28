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
        self.new_vecs = []
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.keys = []
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
        self.analyzer.analyze(scores, self.anchors.nb_anchors)

        self.elite.set_elite(self.param_vecs, self.analyzer)
        elite = self.elite.model

        self.anchors.set_anchors(self.param_vecs, self.analyzer, elite)
        anchors = self.anchors.models
        as_ = [scores[i].item() for i in self.anchors.anchors_idxs]

        self.probes.set_probes(anchors, self.analyzer)
        self.blends.set_blends(anchors, self.param_vecs, self.analyzer)

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
        mylist = []
        # Reset state
        self.shapes = []
        self.num_elems = []
        self.keys = []
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            self.shapes.append(x.size())
            self.num_elems.append(x.numel())
            self.keys.append(key)
            mylist.append(x.reshape(x.numel()))  # Flatten tensor
        vec = torch.cat(mylist)  # Flatten all tensors in model
        return vec

    def construct_pool(self):
        # Define noise magnitude and scale
        print(self.probes.models[0][0:20])
        self.apply_perturbation(self.probes.models)
        self.apply_perturbation(self.blends.models)
        self.new_vecs = []
        self.append_to_list(self.new_vecs, self.anchors.models)
        self.append_to_list(self.new_vecs, self.probes.models)
        self.append_to_list(self.new_vecs, self.blends.models)
        assert len(self.new_vecs) == self.hp.pool_size  # Sanity check

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
        for i, vec in enumerate(self.new_vecs):
            param_list = self.vec_to_tensor(vec)  # Restore shape
            state_dict = self.state_dicts[i]
            self.update_dict(state_dict, param_list)
            # Update model's state dictionary
            self.models[i].load_state_dict(self.state_dicts[i])

    def vec_to_tensor(self, vec):
        a = vec.split(self.num_elems)  # Split parameter tensors
        b = [None]*self.nb_layers
        for i in range(self.nb_layers):
            b[i] = a[i].reshape(self.shapes[i])  # Reconstruct tensor shape
        return b

    def update_dict(self, state_dict, param_list):
        for i, key in enumerate(self.keys):
            state_dict[key] = param_list[i]



#
