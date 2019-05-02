"""Base Class for a Solver. This class contains the different methods that.
"""

from .evaluator import Evaluator
from .interrogator import Interrogator

import torch
import time

class Dataset_Solver(object):
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

    def train_dataset_with_validation(self, iterations):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Training model(s) on a dataset w/ validation")
        self.env.step()
        for _ in range(iterations):
            self.current_iteration += 1
            print ("Iteration %d" %self.current_iteration)
            self.forward()
            self.backward()
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
        self.test()
        self.print_test_accuracy()

    def batch_training(self, iterations):
        """In cases where batch training for the entire dataset is needed."""
        batches = self.env.nb_batches
        for _ in range(iterations):
            self.reset_state()
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                self.env.step()
                self.forward()
                self.backward()
                self.current_batch +=1
            self.current_iteration +=1

    def batch_train_with_validation(self, iterations):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        batches = self.env.nb_batches
        for _ in range(iterations):
            print ("Iteration %d" %self.current_iteration)
            for __ in range(batches):
                self.env.step()
                self.forward()
                self.backward()
                self.current_batch +=1
            self.current_iteration += 1
        self.alg.test()
        self.alg.print_test_accuracy()

    def repeated_batch_train_dataset_with_validation(self, iterations, reps):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        # Local variable definition
        batches = self.env.nb_batches
        self.reset_state()
        # Process
        for _ in range(iterations):
            print ("Iteration %d" %self.current_iteration)
            self.env.step()
            for ___ in range(reps):
                self.forward()
                self.backward()
            self.alg.reset_state()
            self.current_iteration += 1
        self.test()
        self.print_test_accuracy()

    def forward(self):
        self.interrogator.set_inference(self.alg.model, self.env)

    def backward(self):
        self.evaluator.evaluate(self.env, self.interrogator.inference)
        self.alg.step()

    def test(self):
        self.interrogator.get_inference(self.alg.model, self.env, test=True)
        self.evaluator.calculate_correct_predictions(self.interrogator.inference,
                                                     test=True, acc=True)
        self.evaluator.calculate_loss(self.interrogator.inference, test=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_acc = self.evaluator.test_acc
        test_loss = self.evaluator.test_loss  # Assuming minizming loss
        test_loss /= len(self.env.test_data)
        print('Test set: Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
                                                    test_loss, test_acc))

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_batch = 0



#
