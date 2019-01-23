"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

from .hyper_parameters import Hyper_Parameters
from .pool import Pool

class Optimizer:
    def __init__(self, models, hyper_params):
        self.hp = Hyper_Parameters(hyper_params) # Create a hyper parameters object
        self.pool = Pool(models, self.hp) # Create a pool object
        self.integrity = self.hp.initial_integrity

    def inference(self, env):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        outputs = []
        for model in self.pool:
            outputs.append(model(env))
        return outputs

    def calculate_scores(self, outputs):
        """This method calculates the scores based on the given outputs. There
        are many ways to calculate a score, it depends on the type of problem
        being solved.
        Thus, this method can use a second argument, or a hyper parameter, to
        set what type of calculation to use.
        """
        scores = []
        for output in outputs:
            score = output-self.hp.target
            scores.append(score)
        return scores

    def update(self, scores):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        self.set_integrity(scores)
        self.pool.set_new_pool()
        return self.pool.get_pool()






#
