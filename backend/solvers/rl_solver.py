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

    def solve(self, iterations, ep_len=1000):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        self.env.env._max_episode_steps = ep_len
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

    def roll(self):
        steps=0
        obs = self.env.reset_state()
        while not self.env.done:
            self.forward()
            steps+=1

    def forward(self):
        self.interrogator.set_inference(self.alg.model, self.env)
        action = self.interrogator.inference
        self.env.step(action)

    def solve_online_render(self, iterations, ep_len=1000):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        self.env.env._max_episode_steps = ep_len
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            self.env.reset_state()
            #self.alg.reset_state()
            step = 0
            while not self.env.done:
                self.env.render()
                self.forward()
                time.sleep(0.02)  # Delay is 0.03 secs
                self.evaluator.evaluate(self.env, None)
                reward = self.evaluator.score
                feedback = (self.env.observation, self.interrogator.inference,
                            reward)
                if step>10:
                    step = 0
                    self.alg.step(feedback)
                    self.alg.print_state()
                step+=1
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_online(self, iterations, ep_len=1000):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        self.env.env._max_episode_steps = ep_len
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            self.env.reset_state()
            #self.alg.reset_state()
            while not self.env.done:
                self.forward()
                self.evaluator.evaluate(self.env, None)
                reward = self.evaluator.score
                self.alg.step(reward)
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
            reward = 0.
            for _ in range(reps):
                self.roll()
                self.evaluator.evaluate(self.env, None)
                reward += self.evaluator.score
            reward /= reps
            self.alg.step(reward)
            self.alg.print_state()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def solve_aggregator(self, iterations, reps, ep_len=1000):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        self.env.env._max_episode_steps = ep_len
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            print("Episodes: %d" %reps)
            rewards = []
            for _ in range(reps):
                self.roll()
                self.evaluator.evaluate(self.env, None)
                rewards.append(self.evaluator.score)
            reward = self.calc_reward(rewards)
            self.alg.step(reward)
            self.alg.print_state()
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.env.close()

    def calc_reward(self, rewards):
        rewards = torch.Tensor(rewards).cuda()
        var = torch.var(rewards)*0.5
        avg = torch.mean(rewards)
        a = avg-var
        b = rewards.min()
        reward = max(a, b)
        return reward

    def solve_averager_render(self, iterations, reps):
        """In cases where training is needed."""
        print("Training OpenAI environment solver \n")
        for iteration in range(iterations):
            print("Iteration: %d/%d \n" %(iteration, iterations))
            print("Episodes: %d" %reps)
            reward = 0.
            for _ in range(reps):
                self.roll_and_render()
                self.evaluator.evaluate(self.env, None)
                reward += self.evaluator.score
            reward /= reps
            self.alg.step(reward)
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
        self.evaluator.evaluate(self.env, None)
        reward = self.evaluator.score
        print(reward.item())

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
