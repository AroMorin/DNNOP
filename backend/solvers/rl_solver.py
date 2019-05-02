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
import time

class RL_Solver(object):
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
        self.current_iteration = 0

    def solve(self, iterations):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            print("New Episode")
            self.roll()
            self.backward()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_and_render(self, iterations):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            print("New Episode")
            self.roll_and_render()
            self.backward()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def roll(self):
        self.env.reset_state()
        while not self.env.done:
            self.forward()

    def roll_and_render(self, delay=0.05):
        self.env.reset_state()
        while not self.env.done:
            self.env.render()
            self.forward()
            time.sleep(delay)

    def forward(self):
        self.interrogator.set_inference(self.alg.model)
        action = self.interrogator.inference
        self.env.step(action)

    def backward(self):
        self.evaluator.evaluate(self.env, self.interrogator.inference)
        self.alg.step()

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0

    def demonstrate_env(self):
        """In cases where training is needed."""
        self.alg.pool.model = self.alg.pool.elite.model
        self.roll_and_render()
        self.env.close()
