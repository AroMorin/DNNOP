"""A class that defines interfaces to use the SGD solver in PyTorch.
It is expected that the optimizer object will be compatible with the SGD
interface.
"""

from __future__ import division
from .algorithm import Algorithm

import torch
import torch.nn.functional as F
import torch.optim as optim

class SGD(Algorithm):
    def __init__(self, model, alg_params):
        """Model is owned by the class, so it is set as a class attribute.
        Think of the optimizer as the "engine" behind the algorithm.
        """
        print("Using SGD algorithm")
        super(SGD, self).__init__()
        alg_params = self.ingest_params(alg_params)
        self.hyper_params = alg_params
        self.model = model # Model is set as a class attribute
        self.populations = False
        self.optim = None  # The optimizer object
        self.top_score = 10.
        self.minimizing = True
        self.target = 0.
        self.set_optim(alg_params["optimizer"])

    def ingest_params(self, alg_params):
        """Method to loop over the hyper parameter dictionary and update the
        default values.
        """
        default_params = {
                            "learning rate": 0.01,
                            "momentum": 0.5,
                            "optimizer": "SGD",
                            "minimizing": True,
                            "target": 0.
                            }
        default_params.update(alg_params)
        return default_params

    def set_optim(self, optimizer):
        """Method to set the desired optimizer, using the desired hyper
        parameters.
        """
        if optimizer == "SGD":
            self.optim = optim.SGD(self.model.parameters(),
                                        lr = self.hyper_params["learning rate"],
                                        momentum = self.hyper_params["momentum"])
        else:
            print("Unknown optimizer, exiting!")
            exit()

    def step(self, feedback):
        """Implements the main optimization function of the algorithm."""
        inference, score = feedback
        self.top_score = score
        score.backward()
        self.optim.step()
        self.optim.zero_grad()

    def print_state(self):
        print("Train Loss: %f" %self.top_score)



#
