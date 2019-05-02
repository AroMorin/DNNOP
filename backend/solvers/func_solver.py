"""Base Class for a Solver. This class contains the different methods that
"""

from .evaluator import Evaluator
from .interrogator import Interrogator

import torch
import time

class Func_Solver(object):
    """This class makes absolute sense because there are many types of training
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        print("Creating Solver")
        self.env = slv_params['environment']
        self.alg = slv_params['algorithm']
        self.current_iteration = 0
        self.evaluator = Evaluator()
        self.interrogator = Interrogator()

    def solve(self, iterations):
        """In cases where training is needed."""
        print("Training regular solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            self.env.step()
            self.forward()
            self.backward()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def solve_and_plot(self, iterations):
        """In cases where training is needed."""
        print("Training regular solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            self.env.step()
            self.forward()
            self.backward()
            self.env.make_plot(self.alg)
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def forward(self):
        self.interrogator.set_inference(self.alg.model, self.env)

    def backward(self):
        self.evaluator.evaluate(self.env, self.interrogator.inference)
        self.alg.step()



#
