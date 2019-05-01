"""The optimizer class outlines the processes of optimization done by the algorithm.
This is similar to the SGD optimizer in PyTorch, where learning rate and momentum
are specified. In the optimizer class we perform the "update" action of the
weights for the pool.

This class reaches out to Anchors, Blends, Probes, etc..to communicate and
prepare the pool. The pool is the object itself that is being prepared and
updated.
"""

import torch
import torch.nn.functional as F
import time

class Optimizer(object):
    def __init__(self, pool, hyper_params):
        self.pool = pool
        self.hp = hyper_params
        self.env = None

    def set_environment(self, env):
        self.env = env





#
