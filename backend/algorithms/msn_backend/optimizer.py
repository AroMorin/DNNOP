"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

from .msn_backend.hyperparameters import Hyper_Parameters
from .msn_backend.pool import Pool

class Optimizer:
    def __init__(self, pool, hyper_params):
        self.hp = Hyper_Parameters(hyper_params) # Create a hyper parameters object
        self.pool = Pool(pool) # Create a pool object

    def update(self):
        pass
