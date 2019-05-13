"""base class for pool
The pool object will contain the models under optimization.
"""
import torch

class Weights(object):
    def __init__(self, weights):
        self.parameters = weights # Weights dictionary
        self.vector = torch.nn.utils.parameters_to_vector(self.parameters)

    def update(self, vector):
        """It is always assumed that the dict and the vector belong to the same
        model.
        """
        self.vector = vector
        torch.nn.utils.vector_to_parameters(self.vector, self.parameters)
#
