"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

from .msn_backend.hyperparameters import Hyper_Parameters as hp

class Optimizer:
    def __init__(self, hyper_params):
        self.set_hyper_params()




    def update(self):
        pass
