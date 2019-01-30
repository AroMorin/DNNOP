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

import time
import torch

class Pool:
    def __init__(self, models, hyper_params):
        self.models = models # List of Models
        self.hp = hyper_params
        self.analyzer = Analysis(hyper_params)
        self.elite = Elite(hyper_params)
        self.anchors = Anchors(hyper_params)
        self.perturb = Perturbation(hyper_params)
        self.probes = Probes(hyper_params)
        self.blends = Blends(hyper_params)
        self.state_dicts = [] # List of weight dictionaries
        self.vectors = [] # List of parameter vectors
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.keys = []
        self.available_idxs = range(self.hp.pool_size)
        self.idx = None
        self.elite_dict = {}
        self.scores = []
        self.set_state_dicts()
        # Arbitrarily chose 4th model in pool
        self.set_shapes(self.state_dicts[4])
        self.set_vectors()

    def set_state_dicts(self):
        """This method takes in the list of models, i.e. pool, and produces
        a list of weight dictionaries.
        """
        for model in self.models:
            self.state_dicts.append(model.state_dict())
        self.nb_layers = len(model.state_dict())

    def set_shapes(self, dict):
        """We only call this method once since all the pool models are the same
        shape.
        Traverse the dictionary and acquire the shapes.
        """
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            self.shapes.append(x.size())
            self.num_elems.append(x.numel())
            self.keys.append(key)

    def set_vectors(self):
        """This method takes in the list of weight dictionaries and produces
        a list of parameter vectors.
        Note: parameter vectors are essentially "flattened" weights.
        """
        for state_dict in self.state_dicts:
            vec = self.dict_to_vec(state_dict)
            self.vectors.append(vec)

    def dict_to_vec(self, dict):
        mylist = []
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            mylist.append(x.reshape(x.numel()))  # Flatten tensor
        vec = torch.cat(mylist)  # Flatten all tensors in model
        return vec

    def prep_new_pool(self, scores):
        self.reset_state()

        self.analyzer.analyze(scores, self.anchors.nb_anchors)

        self.elite.set_elite(self.vectors, self.analyzer)
        self.anchors.set_anchors(self.vectors, self.analyzer, self.elite.model)
        print("Anchors: ", len(self.anchors.anchors_idxs))
        # Define noise magnitude and scale
        self.perturb.set_perturbation(self.elite.model, self.analyzer)
        self.probes.set_probes(self.anchors, self.perturb)
        self.blends.set_blends(self.anchors, self.vectors, self.analyzer, self.perturb)

        #as_ = [scores[i].item() for i in self.anchors.anchors_idxs]

    def reset_state(self):
        self.available_idxs = range(self.hp.pool_size)
        self.idx = None

    def implement(self):
        """This function updates the ".parameters" of the models using the
        newly-constructed weight dictionaries.
        """
        self.available_idxs = [x for x in self.available_idxs
                                if x not in self.anchors.anchors_idxs]
        self.update_models(self.probes.models)
        self.update_models(self.blends.models)
        assert len(self.available_idxs) == 0  # Sanity

    def update_models(self, vectors):
        for i, vector in enumerate(vectors):
            self.set_idx()
            param_list = self.vec_to_tensor(vector)  # Restore shape
            self.update_dict(param_list)
            # Update model's state dictionary
            self.models[self.idx].load_state_dict(self.state_dicts[self.idx])

    def set_idx(self):
        """This method blindly takes the first available index and loads it into
        the idx attribute. It then proceeds to remove that index from the list
        of available indices.
        """
        self.idx = self.available_idxs[0]
        del self.available_idxs[0]

    def vec_to_tensor(self, vec):
        a = vec.split(self.num_elems)  # Split parameter tensors
        b = [None]*self.nb_layers
        for i in range(self.nb_layers):
            b[i] = a[i].reshape(self.shapes[i])  # Reconstruct tensor shape
        return b

    def update_dict(self, param_list):
        state_dict = self.state_dicts[self.idx]
        for i, key in enumerate(self.keys):
            state_dict[key] = param_list[i]









#
