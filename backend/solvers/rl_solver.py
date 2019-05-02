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

class RL_Solver(object):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, env, algorithm):
        print("Creating Solver")
        self.evaluator = Evaluator()
        self.interrogator = Interrogator()
        self.current_iteration = 0
        self.current_step = 0

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

    def roll_and_render(self):
        self.env.reset_state()
        while not self.env.done:
            if self.env.rendering:
                self.env.render()
            self.forward()

    def forward(self):
        self.interrogator.set_inference(self.alg.model)
        action = self.interrogator.inference
        self.env.step(action)

    def backward(self):
        self.evaluator.evaluate()
        self.alg.step()


    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0
        self.current_step = 0

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
