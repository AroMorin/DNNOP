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
from .sar_backend.hyper_parameters import Hyper_Parameters
from .sar_backend.engine import Engine

class SAR(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using SAR algorithm")
        super(SAR, self).__init__()
        self.model = model
        self.hyper_params = Hyper_Parameters(alg_params)
        self.engine = Engine(self.hyper_params)
        self.populations = False
        self.grad = False
        self.minimizing = self.hyper_params.minimizing
        self.initial_score = self.hyper_params.initial_score
        self.top_score = self.initial_score
        self.target = None
        self.set_target()

    def step(self, feedback):
        """This method takes in the environment, runs the models against it,
        obtains the scores and accordingly updates the models.
        """
        _, _, score = feedback
        print(score)
        self.engine.analyze(feedback, self.top_score)
        self.engine.update_state()
        self.engine.update_table(self.model, feedback)
        self.update_score(score)

    def update_score(self, score):
        self.top_score = score

    def update_top_score_(self, score):
        """Analysis is still needed even if there's no improvement,
        so other modules know that this as well. Hence, can't "return" after
        initial condition.
        """
        if self.engine.jumped:
            self.top_score = score
        else:
            v = 0.0000
            if self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.+v)
            elif self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.+v)

    def print_state(self):
        print ("Score: %f" %self.top_score)
        print("Intrinsic Reward: %f" %self.engine.analyzer.intrinsic_reward.value)

#
