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
        self.current_iteration = 0
        self.current_batch = 0
        self.current_step = 0

    def solve(self, iterations):
        """In cases where training is needed."""
        print("Training regular function solver \n")
        # Local variable definition
        env = self.env
        alg = self.algorithm
        for iteration in range(iterations):
            print("Iteration: %d\n" %iteration)
            env.step()
            alg.optimize(env)
            if env.plot:
                env.make_plot(alg)
            self.current_iteration +=1
            print("\n")
            if alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def batch_training(self, epochs):
        """In cases where batch training is needed."""
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches
        self.reset_state()

        for _ in range(epochs):
            for __ in range(batches):
                env.step()
                alg.optimize(env)
                self.current_iteration +=1

    def train_dataset_with_validation(self, steps):
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
        for _ in range(steps):
            self.current_iteration += 1
            print ("Iteration %d" %self.current_iteration)
            alg.optimize(env)
            alg.test(env)
            alg.print_test_accuracy(env)
            if alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization

    def batch_train_dataset_with_validation(self, steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches

        # Process
        for _ in range(steps):
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                print("Batch %d" %self.current_batch)
                env.step()
                alg.optimize(env)
                self.current_batch +=1
            alg.test(env)
            alg.print_test_accuracy(env)
            self.current_iteration += 1

    def repeated_batch_train_dataset_with_validation(self, steps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches
        patience = 16
        self.reset_state()

        # Process
        for _ in range(steps):
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                print("Batch %d" %self.current_batch)
                env.step()
                self.current_step = 0  # Reset step count
                for ___ in range(patience):
                    print("Step %d" %self.current_step)
                    alg.optimize(env)
                    self.current_step += 1
                self.current_batch +=1
            alg.test(env)
            alg.print_test_accuracy(env)
            self.current_iteration += 1

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_iteration = 0
        self.current_batch = 0
        self.current_step = 0
