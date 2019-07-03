"""Base Class for a Solver. This class contains the different methods that.
"""
from .solver import Solver

import torch
import time, csv

class Dataset_Solver(Solver):
    """This class makes absolute sense because there are many types of training
    the user -which is the ultimate goal, complete transparency-.
    """
    def __init__(self, slv_params):
        super(Dataset_Solver, self).__init__(slv_params)
        self.current_iteration = 0
        self.scores = []

    def train_dataset_with_validation(self, iterations):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Training model(s) on a dataset w/ validation")
        self.env.step()
        for _ in range(iterations):
            self.current_iteration += 1
            print ("\nIteration %d" %self.current_iteration)
            self.forward()
            self.backward()
            if self.alg.achieved_target():
                print ("Achieved/exceeded target")
                break # Terminate optimization
            self.log()
        self.test()
        self.print_test_accuracy()
        self.save_data()

    def batch_training(self, iterations):
        """In cases where batch training for the entire dataset is needed."""
        batches = self.env.nb_batches
        for _ in range(iterations):
            self.reset_state()
            print ("\nIteration %d" %self.current_iteration)
            for __ in range(batches):
                self.env.step()
                self.forward()
                self.backward()
            self.current_iteration +=1

    def batch_train_with_validation(self, iterations):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        batches = self.env.nb_batches
        for _ in range(iterations):
            print ("\nIteration %d" %self.current_iteration)
            for __ in range(batches):
                self.env.step()
                self.forward()
                self.backward()
            self.current_iteration += 1
        self.alg.test()
        self.alg.print_test_accuracy()

    def determined_batch_train_with_validation(self, iterations):
        """In cases where a dataset is being trained with a validation component
        such as MNIST.
        """
        print("Mini-batch training model(s) on a dataset w/ validation")
        batches = self.env.nb_batches
        for _ in range(iterations):
            print ("\nIteration %d" %self.current_iteration)
            for __ in range(batches):
                improved = False
                self.alg.reset_state()
                self.env.step()
                #self.forward()
                #self.backward()
                step=0
                while step<1000:
                    print("Batch: %d" %__)
                    self.forward()
                    self.backward()
                    improved = self.alg.engine.analyzer.improved
                    step+=1
            self.current_iteration += 1
        self.test()
        self.print_test_accuracy()

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
            print ("\nIteration %d" %_)
            self.env.step()
            for __ in range(reps):
                print ("\nRep %d" %__)
                self.forward()
                self.backward()
            self.alg.reset_state()
        self.test()
        self.print_test_accuracy()

    def test(self):
        self.alg.eval()
        self.interrogator.set_inference(self.alg.model, self.env, test=False)
        self.evaluator.calculate_loss(self.env, self.interrogator.inference,
        test=False)
        self.evaluator.calculate_correct_predictions(self.env,
                                                    self.interrogator.inference,
                                                     test=False, acc=True)
        self.interrogator.set_inference(self.alg.model, self.env, test=True)
        self.evaluator.calculate_loss(self.env, self.interrogator.inference,
        test=True)
        self.evaluator.calculate_correct_predictions(self.env,
                                                    self.interrogator.inference,
                                                     test=True, acc=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        train_acc = self.evaluator.train_acc
        train_loss = self.evaluator.train_loss  # Assuming minizming loss
        print('Train set: Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
                                                    train_loss, train_acc))
        test_acc = self.evaluator.test_acc
        test_loss = self.evaluator.test_loss  # Assuming minizming loss
        print('Test set: Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
                                                    test_loss, test_acc))

    def reset_state(self):
        """This is probably in cases of RL and such where an "envrionment"
        can be reset.
        """
        self.current_batch = 0

    def log(self):
        self.scores.append(self.alg.top_score.item())

    def save_data(self):
        self.steps = list(range(len(self.scores)))
        with open('train_losses.csv', mode='w') as data_file:
            data_writer = csv.writer(data_file)
            for i in self.steps:
                data_writer.writerow((i, self.scores[i]))
        data_file.close()

#
