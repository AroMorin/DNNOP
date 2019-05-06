"""Base Class for a Solver. This class contains the different methods that
can be used to solve an environment/problem. There are methods for
mini-batch training, control, etc...
The idea is that this class will contain all the methods that the different
algorithms would need. Then we can simply call this class in the solver scripts
and use its methods.
I'm still torn between using a class or just using a script.
"""
from .evaluator import Evaluator
from .interrogator import Interrogator

import torch

class Solver(object):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        print("Creating Solver")
        self.env = slv_params['environment']
        self.alg = slv_params['algorithm']
        self.evaluator = Evaluator()
        self.interrogator = Interrogator()

    def forward(self):
        self.interrogator.set_inference(self.alg.model, self.env)

    def backward(self):
        self.evaluator.evaluate(self.env, self.interrogator.inference)
        feedback = (self.interrogator.inference, self.evaluator.score)
        self.alg.step(feedback)
        self.alg.print_state()

    def save(self, path=''):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        torch.save(self.alg.model.state_dict(), fn)

    def save_pool_weights(self, models, path):
        for i, model in enumerate(models):
            fn = path+"model_"+str(i)+".pth"
            torch.save(model.state_dict(), fn)

    def save_elite_weights(self, model, path):
        fn = path+"model_elite.pth"
        torch.save(model.state_dict(), fn)

    def load(self, path):
        """Only works with my algorithms, not with SGD."""
        fn = path+"model_elite.pth"
        self.alg.model.load_state_dict(torch.load(fn))








#
