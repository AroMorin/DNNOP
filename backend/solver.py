"""Base Class for a Solver. This class contains the different methods that
can be used to solve an environment/problem. There are methods for
mini-batch training, control, etc...
The idea is that this class will contain all the methods that the different
algorithms would need. Then we can simply call this class in the solver scripts
and use its methods.
I'm still torn between using a class or just using a script.
"""

import torch
import time
from solvers.rl_solvers import RL_Solver
from solvers.func_solvers import Func_Solver
from solvers.dataset_solvers import Dataset_Solver

class Solver(object):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        print("Creating Solver")
        slv_params = self.ingest_params(slv_params)
        self.init_solver(slv_params)

    def ingest_params(self, slv_params):
        default_params = {
                            "problem type": '',
                            'environment': None,
                            'algorithm': None
                            }
        default_params.update(slv_params)
        return default_params

    def init_solver(self, slv_params):
        if slv_params['problem type'] == 'dataset':
            self.dataset_solver = Dataset_Solver(slv_params)
        elif slv_params['problem type'] == 'RL':
            self.rl_solver = RL_Solver(slv_params)
        elif slv_params['problem type'] == 'function':
            self.func_solver = Func_Solver(slv_params)
        else:
            print("Unknown solver requested, exiting!")
            exit()

    def save(self, path=''):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        torch.save(self.alg.model.state_dict(), fn)

    def load(self, path):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        self.alg.model.load_state_dict(torch.load(fn))









#
