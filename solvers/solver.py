"""Base Class for a Solver"""

import torch

class Solver():
    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm
        self.current_epoch = ''
        self.current_step = ''

    def batch_training(self, epochs):
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches

        # Process
        for epoch in range(epochs):
            for batch in range(batches):
                env.step()
                alg.optimize(env)
                self.current_step +=1

    def train_dataset_with_validation(self, epochs):
        # Local variable definition
        env = self.env
        alg = self.algorithm
        batches = self.env.nb_batches

        # Process
        for epoch in range(epochs):
            for batch in range(batches):
                env.step()
                alg.optimize(env)
                self.current_step +=1
            alg.test(env)
            alg.get_test_accuracy(env)
            self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0
        self.current_step = 0
