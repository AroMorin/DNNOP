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
from .learner_backend.hyper_parameters import Hyper_Parameters
from .learner_backend.engine import Engine

class LEARNER(Algorithm):
    def __init__(self, model, alg_params):
        print ("Using Learner algorithm")
        super(LEARNER, self).__init__()
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
        print(score)
        #score = self.regularize(score)
        self.engine.analyze(score, self.top_score)
        self.engine.set_elite()
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
            v = 0.00005
            if self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.+v)
            elif self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score>0.:
                self.top_score = self.top_score*(1.-v)
            elif not self.minimizing and self.top_score<0.:
                self.top_score = self.top_score*(1.+v)

    def print_state(self):
        print ("Top Score: %f" %self.top_score)

    def print_state_(self):
        if self.engine.analyzer.replace:
            print ("------Setting new Elite-------")
        if self.engine.frustration.jump:
            print("------WOOOOOOHHOOOOOOOO!-------")
        if self.engine.analyzer.improved:
            print("Improved!")
        print ("Top Score: %f" %self.top_score)
        print("Memory: %d" %self.engine.frustration.count)
        print("Jump: %f" %(100.*self.engine.frustration.value))
        print("Integrity: %f" %self.engine.integrity.value)
        print("Bin: ", self.engine.integrity.step_size.bin)
        print("Step size: %f" %self.engine.integrity.step_size.value)
        print("SR: (%f, %f)" %(self.engine.noise.sr_min, self.engine.noise.sr_max))
        print("Selections: %d" %self.engine.noise.num_selections)
        print("V: ", self.engine.elite[0:15])
        print("Distance: %f" %self.engine.diversity.min_distance)

    def eval(self):
        self.engine.vector = self.engine.elite
        self.engine.update_weights(self.model)

#
