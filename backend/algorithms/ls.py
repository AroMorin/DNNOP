"""The Class for the Local Search algorithm.
"""
from __future__ import division
from .algorithm import Algorithm
from .ls_backend.hyper_parameters import Hyper_Parameters
from .ls_backend.engine import Engine

class LS(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using Local Search algorithm")
        super(LS, self).__init__()
        self.hyper_params = Hyper_Parameters(alg_params) # Create a hyper parameters object
        self.engine = Engine(model, self.hyper_params) # Create a pool object
        self.populations = False
        self.model = model
        self.minimizing = self.hyper_params.minimizing
        self.initial_score = self.hyper_params.initial_score
        self.top_score = self.initial_score
        self.target = None
        self.set_target()

    def step(self, feedback):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        score = feedback
        score = score.detach()
        #score = self.regularize(score)
        self.engine.analyze(score, self.top_score)
        self.engine.set_elite()
        self.engine.set_vector()
        self.engine.update_state()
        self.engine.generate()
        self.engine.update_weights(self.model)
        self.update_top_score(score)

    def regularize(self, score):
        norm = abs(self.engine.vector.mean())
        penalty = norm
        print("Regularization: %f" %penalty.item())
        if self.minimizing and score>0.:
            score = score+penalty
        elif self.minimizing and score<0.:
            score = score+penalty
        elif not self.minimizing and score>0.:
            score = score-penalty
        elif not self.minimizing and score<0.:
            score = score-penalty
        return score

    def update_top_score_(self, score):
        """Analysis is still needed even if there's no improvement,
        so other modules know that this as well. Hence, can't "return" after
        initial condition.
        """
        if self.engine.jumped:
            self.top_score = score
        else:
            #v = 0.00002
            v = 0.002
            #v = 0.0
            if self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.+v)
            elif self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.+v)

    def update_top_score(self, score):
        """Analysis is still needed even if there's no improvement,
        so other modules know that this as well. Hence, can't "return" after
        initial condition.
        """
        if self.engine.jumped:
            self.top_score = score
        else:
            #v = 0.00002
            v = 0.005
            #v = 0.0
            if self.minimizing:
                self.top_score = self.top_score+v
            elif not self.minimizing:
                self.top_score = self.top_score-v

    def print_state_(self):
        print ("Top Score: %f" %self.top_score)

    def print_state(self):
        if self.engine.analyzer.replace:
            print ("------Setting new Elite-------")
        if self.engine.analyzer.improved:
            print("Improved!")
        print ("Top Score: %f" %self.top_score)
        print("LR: %f" %self.engine.noise.sr.lr)

    def eval(self):
        self.engine.vector = self.engine.elite
        self.engine.update_weights(self.model)

#
