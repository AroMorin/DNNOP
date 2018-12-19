"""main script
Its main functions are:
1) Init a pool object
2) Evaluate the models
3) Get/Set the model weights
4) Get the new pool
"""
import torch
from .advanced_neuroevolution_backend import optimizer
from .algorithm import Algorithm

class Advanced_Neuroevolution(Algorithm):
    def __init__(self, model):
        super().__init__(model)
