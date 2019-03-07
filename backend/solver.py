"""Base Class for a Solver. This class contains the different methods that
can be used to solve an environment/problem. There are methods for
mini-batch training, control, etc...
The idea is that this class will contain all the methods that the different
algorithms would need. Then we can simply call this class in the solver scripts
and use its methods.
I'm still torn between using a class or just using a script.
"""

import torch

class Solver(object):
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, env, algorithm):
        print("Creating Solver")
        self.env = env
        self.algorithm = algorithm
        self.current_iteration = 0
        self.current_batch = 0
        self.current_step = 0

    def solve(self, iterations):
        """In cases where training is needed."""
        print("Training regular function solver \n")
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            self.env.step()
            self.alg.optimize(self.env)
            if self.env.plot:
                self.env.make_plot(self.alg)
            self.current_iteration +=1
            print("\n")
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def batch_training(self, epochs):
        """In cases where batch training is needed."""
        batches = self.env.nb_batches
        self.reset_state()
        for _ in range(epochs):
            for __ in range(batches):
                self.env.step()
                self.alg.optimize(env)
                self.current_iteration +=1

    def train_dataset_with_validation(self, steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        Note: the names of the functions called here have to be universal among
        algorithms. This ensures the desired "plug n play" functionality.
        """
        print("Training model(s) on a dataset w/ validation")
        self.env.step()
        for _ in range(steps):
            self.current_iteration += 1
            print ("Iteration %d" %self.current_iteration)
            self.alg.optimize(self.env)
            self.alg.test(self.env)
            self.alg.print_test_accuracy(self.env)
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def batch_train_dataset_with_validation(self, steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        # Local variable definition
        batches = self.env.nb_batches

        # Process
        for _ in range(steps):
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                print("Batch %d" %self.current_batch)
                self.env.step()
                self.alg.optimize(self.env)
                self.current_batch +=1
            self.alg.test(self.env)
            self.alg.print_test_accuracy(self.env)
            self.current_iteration += 1

    def repeated_batch_train_dataset_with_validation(self, steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        # Local variable definition
        batches = self.env.nb_batches
        reps = 16  # repititions
        self.reset_state()

        # Process
        for _ in range(steps):
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                print("Batch %d" %self.current_batch)
                self.env.step()
                self.current_step = 0  # Reset step count
                for ___ in range(reps):
                    print("Step %d" %self.current_step)
                    self.alg.optimize(self.env)
                    self.current_step += 1
                self.current_batch +=1
            self.alg.test(self.env)
            self.alg.print_test_accuracy(self.env)
            self.current_iteration += 1

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0
        self.current_batch = 0
        self.current_step = 0