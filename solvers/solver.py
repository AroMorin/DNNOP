"""Base Class for a Solver. I'm not really sure why I put this class
in here instead of in the backend folder.
"""

import torch

class Solver():
    """This class makes absolute sense because there are many types of training
    depending on the task. For this reason, in the future, this class can easily
    include all instances of such training routines. Of course, transparent to
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, env, algorithm):
        print("Creating Solver")
        self.env = env
        self.algorithm = algorithm
        self.current_step = 0
        self.elapsed_steps = 0

    def batch_training(self, epochs):
        """In cases where batch training is needed."""
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches

        # Process
        for _ in range(epochs):
            for __ in range(batches):
                env.step()
                alg.optimize(env)
                self.elapsed_steps +=1

    def train_dataset_with_validation(self, optimization_steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        Note: the names of the functions called here have to be universal among
        algorithms. This ensures the desired "plug n play" functionality.
        """
        print("Training model(s) on a dataset w/ validation")
        # Local variable definition
        env = self.env
        alg = self.algorithm

        # Process
        env.step()
        for _ in range(optimization_steps):
            self.elapsed_steps += 1
            print ("Iteration %d" %self.elapsed_steps)
            alg.optimize(env)
            alg.test(env)
            alg.print_test_accuracy(env)
            if alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def batch_train_dataset_with_validation(self, optimization_steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Training model(s) on a dataset w/ validation")
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches

        # Process
        for _ in range(optimization_steps):
            for __ in range(batches):
                env.step()
                alg.optimize(env)
                self.current_step +=1
            alg.test(env)
            alg.print_test_accuracy(env)
            self.elapsed_steps += 1

    def reset(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_epoch = 0
        self.current_step = 0
