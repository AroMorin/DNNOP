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

class Pool(object):
    def __init__(self, model, hyper_params):
        self.model = model
        self.hp = hyper_params
        self.analyzer = Analysis(hyper_params)
        self.elite = Elite(hyper_params)
        self.anchors = Anchors(hyper_params)
        self.perturb = Perturbation(hyper_params)
        self.probes = Probes(hyper_params)
        self.blends = Blends(hyper_params)
        self.state_dict = {} # Weights dictionary
        self.vector = None # Parameter vector
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.keys = []
        self.current_anchor = 0
        self.nb_probes = 0  # State
        self.nb_blends = 0  # State
        self.new_gen = True
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
        self.analyzer.analyze(score, self.anchors.nb_anchors)
        self.score = self.analyzer.score
        self.elite.set_elite(self.model, self.inference, self.score)
        self.anchors.set_anchors(self.vector, self.inference, self.score)

        # Define noise magnitude and scale
        self.perturb.update_state(self.analyzer)
        self.blends.update_state(self.anchors, self.analyzer, self.perturb)

    def generate(self):
        self.new_gen = False  # Update state
        self.set_next()
        if self.next == "probe":
            self.probes.generate(self.anchors.vectors[self.current_anchor], self.perturb)
            self.vector = self.probes.vector
        elif self.next == "blend":
            self.blends.generate(self.vector)
            self.vector = self.blends.vector
        else:
            print("unknown generate type, exiting!")
            exit()
        self.update_model(self.vector)

    def set_next(self):
        if self.nb_probes < self.hp.nb_probes:
            self.next = "probe"
            self.nb_probes+=1  # Increment probe count
        else:
            if self.current_anchor < (self.anchors.nb_anchors-1):
                self.current_anchor+=1  # Move to next Anchor
                self.nb_probes = 0  # Reset probe count
                self.next = "probe"
            else:
                if self.nb_blends < self.blends.nb_blends:
                    self.next = "blend"  # No more anchors, moving to blends
                    self.nb_blends+=1  # Increment blend count
                else:
                    self.reset_state()
        print(self.next)

    def reset_state(self):
        """Updates the state of the class."""
        print("----------New Optimization Generation--------")
        self.new_gen = True
        self.next = "probe"
        self.current_anchor = 0  # Reset Anchors
        self.nb_probes = 0
        self.nb_blends = 0

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
