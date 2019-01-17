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

    def inference(self, env):
        """This method runs inference on the given environment using the models.
        """
        # return scores
        pass

    def update(self, scores):
        """This method takes in the scores, feeds it to the pool so that the
        selection and update process can occur.
        The pool thus updates itself.
        """
        pass
