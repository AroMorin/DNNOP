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
from .evaluator import Evaluator
from .interrogator import Interrogator
from solvers.rl_solvers import RL_Solver
from solvers.dataset_solvers import Dataset_Solver
from solvers.func_solvers import Func_Solver

class Solver(object):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, env, algorithm):
        print("Creating Solver")
        self.env = env
        self.alg = algorithm
        self.evaluator = Evaluator()
        self.interrogator = Interrogator()
        self.rl_solver = RL_Solver()
        self.current_iteration = 0
        self.current_batch = 0
        self.alg.set_environment(self.env)

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0
        self.current_batch = 0

    def save(self, path=''):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        torch.save(self.alg.pool.elite.model.state_dict(), fn)

    def load(self, path):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        self.alg.pool.elite.model.load_state_dict(torch.load(fn))

    def demonstrate_env(self):
        """In cases where training is needed."""
        self.alg.env.reset_state()
        self.alg.pool.model = self.alg.pool.elite.model
        while not self.alg.env.done:
            self.alg.env.render()
            self.alg.get_inference()
            action = self.alg.inference
            self.alg.env.step(action)
            time.sleep(0.05)
        self.alg.env.close()









#
