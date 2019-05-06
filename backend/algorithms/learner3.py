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
from .learner3_backend.engine import Engine
import time

class LEARNER3(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using Learner3 algorithm")
        super(LEARNER3, self).__init__()
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.engine = Engine(model, self.hyper_params) # Create a pool object
        self.populations = False
        self.model = model

    def optimize(self, observation, inference, score):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        self.engine.elite.set_elite(self.model, self.engine.weights.vector, inference, score)
        self.engine.prep_new_model(observation, inference, score)
        self.engine.generate()

    def print_state(self):
        print("Score: %f" %self.score.item())
        if self.elite.replace:
            print ("------Setting new Elite-------")
        if self.integrity.improvement:
            print("Improved!")
        print ("Elite Score: %f" %self.elite_score)
        print("Integrity: %f" %self.analyzer.integrity)
        print("Steps to Backtrack: %d" %(self.hp.patience-self.analyzer.elapsed_steps+2))
        print(self.analyzer.bin)
        print(self.analyzer.step_size)
        print("SR: %f" %self.analyzer.search_radius)
        print("Selections(%%): %f" %self.analyzer.num_selections)
        print("Selections: %d" %self.perturb.size)
        print("P: ", self.perturb.p[0:10])
        print("Variance(P): %f" %self.perturb.variance)





#
