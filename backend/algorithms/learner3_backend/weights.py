"""base class for pool
The pool object will contain the models under optimization.
"""
import torch

class Weights(object):
    def __init__(self, weights):
        self.current = weights # Weights dictionary
        self.nb_layers = 0
        self.shapes = []
        self.num_elems = []
        self.keys = []
        self.vector = None
        self.set_shapes(self.current)
        self.set_vector()

    def update(self, weights, mode='vec2dict'):
        """It is always assumed that the dict and the vector belong to the same
        model.
        """
        self.vector = weights
        self.vec_to_dict()

    def set_shapes(self, dict):
        """We only call this method once since all the pool models are the same
        shape.
        Traverse the dictionary and acquire the shapes.
        """
        self.nb_layers = len(dict)
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            self.shapes.append(x.size())
            self.num_elems.append(x.numel())
            self.keys.append(key)

    def set_vector(self):
        """Changes the dictionary of weights into a vector."""
        dict = self.current
        mylist = []
        for i, key in enumerate(dict):
            x = dict[key]  # Get tensor of parameters
            mylist.append(x.reshape(x.numel()))  # Flatten tensor
        self.vector = torch.cat(mylist)  # Flatten all tensors in model

    def vec_to_dict(self):
        """Updates the weight dictionaries of the models."""
        param_list = self.vec_to_list()  # Restore shape
        self.update_dict(param_list)

    def vec_to_list(self):
        """Changes a vector into a tensor using the original network shapes."""
        a = self.vector.split(self.num_elems)  # Split parameter tensors
        b = [None]*self.nb_layers
        for i in range(self.nb_layers):
            b[i] = a[i].reshape(self.shapes[i])  # Reconstruct tensor shape
        return b

    def update_dict(self, param_list):
        """Updates the state dictionary class attribute."""
        for i, key in enumerate(self.keys):
            self.current[key] = param_list[i]


#
