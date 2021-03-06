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
from .rs_backend.hyper_parameters import Hyper_Parameters
from .rs_backend.engine import Engine

class RS(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using Learner algorithm")
        super(RS, self).__init__()
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
            score = score-penalty
        elif not self.minimizing and score>0.:
            score = score-penalty
        elif not self.minimizing and score<0.:
            score = score+penalty
        return score

    def update_top_score_(self, score):
        """Analysis is still needed even if there's no improvement,
        so other modules know that this as well. Hence, can't "return" after
        initial condition.
        """
        self.top_score = score

    def update_top_score(self, score):
        """Analysis is still needed even if there's no improvement,
        so other modules know that this as well. Hence, can't "return" after
        initial condition.
        """
        if self.engine.jumped:
            self.top_score = score
        else:
            v = 0.0000
            #v = 0.0002
            #v = 0.05
            if self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.+v)
            elif self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.+v)

    def print_state_(self):
        print ("Top Score: %f" %self.top_score)

    def print_state(self):
        if self.engine.analyzer.replace:
            print ("------Setting new Elite-------")
        if self.engine.analyzer.improved:
            print("Improved!")
        print ("Top Score: %f" %self.top_score)

    def eval(self):
        self.engine.vector = self.engine.elite
        self.engine.update_weights(self.model)

#
