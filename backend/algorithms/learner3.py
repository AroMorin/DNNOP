"""It is expected that the hyper_params object passed to the class is compatible
with the chosen algorithm. Thus, since Learner is chosen here, it is expected that
the hyper_params object will contain the expected information/params in the
expected locations.

We need to create an optimizer object. This object will be initialized with the
desired hyper parameters. An example of hyper params is the number of Anchors.
The optimizer object will own the pool.?
"""
from __future__ import division
from .algorithm import Algorithm
import torch
import numpy as np
from .learner3_backend.hyper_parameters import Hyper_Parameters
from .learner3_backend.pool import Pool
from .learner3_backend.optimizer import Optimizer
import time

class LEARNER3(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using Learner3 algorithm")
        super(LEARNER3, self).__init__()
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.optim = Optimizer(model, self.hyper_params) # Create a pool object
        self.populations = False
        self.model = model

    def optimize(self):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.interrogator.get_inference(self.model)
        self.evaluator.evaluate(self.inference)
        self.evaluator.print_score()
        self.optim.step()









#
