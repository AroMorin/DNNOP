"""Base Class for a Solver. This class contains the different methods that
can be used to solve an environment/problem. There are methods for
mini-batch training, control, etc...
The idea is that this class will contain all the methods that the different
algorithms would need. Then we can simply call this class in the solver scripts
and use its methods.
I'm still torn between using a class or just using a script.
"""

from .solver import Solver

import torch
import time

class RL_Solver(Solver):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        super(RL_Solver, self).__init__(slv_params)
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

    def roll(self, silent=False):
        steps=0
        obs = self.env.reset_state()
        while not self.env.done:
            self.forward()
            steps+=1
        if not silent:
            print("Took %d steps" %steps)

    def forward(self):
        self.interrogator.set_inference_chain(self.alg.model, self.env)
        action = self.interrogator.inference
        self.env.step(action)

    def solve_online_render(self, iterations):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            #self.roll(silent=True)
            self.env.reset_state()
            while not self.env.done:
                self.env.render()
                self.forward()
                time.sleep(0.03)  # Delay is 0.03 secs
                self.evaluator.evaluate(self.env, self.interrogator.inference)
                feedback = self.evaluator.score
                self.alg.step(feedback)
                self.alg.print_state()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_online(self, iterations):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            #self.roll(silent=True)
            self.env.reset_state()
            while not self.env.done:
                self.forward()
                self.evaluator.evaluate(self.env, self.interrogator.inference)
                feedback = self.evaluator.score
                self.alg.step(feedback)
                self.alg.print_state()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_averager(self, iterations, reps, ep_len=1000):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        self.env.env._max_episode_steps = ep_len
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            print("Episodes: %d" %reps)
            feedback = 0.
            for _ in range(reps):
                #self.roll(silent=True)
                self.roll()
                self.evaluator.evaluate(self.env, self.interrogator.inference)
                feedback += self.evaluator.score
            feedback /= reps
            self.alg.step(feedback)
            self.alg.print_state()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_averager_render(self, iterations, reps):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            print("Episodes: %d" %reps)
            feedback = 0.
            for _ in range(reps):
                #self.roll(silent=True)
                self.roll_and_render()
                self.evaluator.evaluate(self.env, self.interrogator.inference)
                feedback += self.evaluator.score
            feedback /= reps
            self.alg.step(feedback)
            self.alg.print_state()
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

    def roll_and_render(self, delay=0.03):
        self.env.reset_state()
        while not self.env.done:
            self.env.render()
            self.forward()
            time.sleep(delay)

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0

    def demonstrate_env(self, episodes=1, ep_len=1000):
        """In cases where training is needed."""
        self.env.env._max_episode_steps = ep_len
        self.alg.eval()
        for _ in range(episodes):
            self.roll_and_render()
        self.env.close()
