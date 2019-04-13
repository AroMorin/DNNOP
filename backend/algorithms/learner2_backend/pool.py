"""base class for pool
Its functions are:
1) Initializing the pool with the given random conditions
2) Returning/Setting the pool
3) Sorting the pool members according to their performance
4) Maintain the pool composition of Elite, Anchors and Probes


The pool object will contain the models under optimization.
"""
from .elite import Elite
from .probes import Probes
from .analysis import Analysis
from .perturbation import Perturbation


import time
import torch

class Pool(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.hp = hyper_params
        self.analyzer = Analysis(hyper_params)
        self.elite = Elite(hyper_params)
        self.perturb = Perturbation(hyper_params)
        self.probes = Probes(hyper_params)
        self.state_dict = {} # Weights dictionary
        self.vector = None # Parameter vector
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.keys = []
        self.score = self.hp.initial_score
        self.set_state_dict()
        self.set_shapes(self.state_dict)
        self.set_vector()
        self.perturb.init_perturbation(self.vector)

    def set_state_dict(self):
        """This method takes in the list of models, i.e. pool, and produces
        a list of weight dictionaries.
        """
        self.state_dict = self.model.state_dict()
        self.nb_layers = len(self.state_dict)

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

    def set_vector(self):
        """Changes the dictionary of weights into a vector."""
        dict = self.state_dict
        mylist = []
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            mylist.append(x.reshape(x.numel()))  # Flatten tensor
        self.vector = torch.cat(mylist)  # Flatten all tensors in model

    def prep_new_model(self, inference, score):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.inference = inference
        self.analyzer.analyze(score)
        self.score = self.analyzer.score
        self.elite.set_elite(self.model, self.vector, self.inference, self.score,
                            self.analyzer.improvement)
        # Define noise magnitude and scale
        self.perturb.update_state(self.analyzer)

    def generate(self):
        self.probes.generate(self.elite.vector, self.perturb)
        self.vector = self.probes.vector
        self.update_model(self.vector)
        #time.sleep(0.5)

    def update_model(self, vector):
        """Updates the weight dictionaries of the models."""
        param_list = self.vec_to_tensor(vector)  # Restore shape
        self.update_dict(param_list)
        # Update model's state dictionary
        self.model.load_state_dict(self.state_dict)

    def vec_to_tensor(self, vec):
        """Changes a vector into a tensor using the original network shapes."""
        a = vec.split(self.num_elems)  # Split parameter tensors
        b = [None]*self.nb_layers
        for i in range(self.nb_layers):
            b[i] = a[i].reshape(self.shapes[i])  # Reconstruct tensor shape
        return b

    def update_dict(self, param_list):
        """Updates the state dictionary class attribute."""
        for i, key in enumerate(self.keys):
            self.state_dict[key] = param_list[i]








#
