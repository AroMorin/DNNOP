"""main script
Its main functions are:
1) Init a pool object
2) Evaluate the models
3) Get/Set the model weights
4) Get the new pool
"""
import torch
import torch.nn.functional as F
from .msn_backend import optimizer
from .algorithm import Algorithm

class MSN(Algorithm):
    def __init__(self, nb_samples):
        super().__init__(nb_samples)
        self.optimizer = optimizer()
        self.nb_samples = ''
        self.anchors = ''
        self.probes = ''

    def optimize(self, env):
        pass
