"""Class for analysis operations on the scores."""

from __future__ import division
from .integrity import Integrity
import torch
import math
import time

class Analysis(object):
    def __init__(self, hyper_params):
        self.hp = hyper_params
        self.integrity = Integrity(hyper_params)
        self.num_selections = None
        self.search_radius = None

    def analyze(self, score):
        """The main function."""
        self.integrity.set_integrity(score)
        self.set_num_selections()
        self.set_search_radius()
        self.update_state()

    def set_num_selections(self):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        #p = 1-self.integrity
        p = self.integrity.value
        numerator = self.hp.alpha
        denominator = 1+(self.hp.beta/p)
        self.num_selections = numerator/denominator

    def set_search_radius(self):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = 1-self.integrity.value
        argument = (self.hp.lambda_*p)-2.5
        exp1 = math.tanh(argument)+1
        self.search_radius = exp1*self.hp.lr














#
