"""This script attempts to solve the classification problem of the MNIST
dataset. Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import environments
import backend.models as models
import backend.algorithms as algorithms
import torch.nn.functional as F
from solver import Solver
#from comet_ml import Experiment

import argparse
import torch
import torch.optim as optim

def main():
    # Assumes CUDA is available
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    # Make an MNIST Dataset environment
    data_path = "C:/Users/aaa2cn/Documents/mnist_data"
    env = environments.make("dataset", "mnist", args.batch_size, data_path)

    # Make a model
    precision = torch.float
    model = models.make("MNIST CNN", precision)

    # Make an algorithm --algorithm owns the model--
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    sgd = algorithms.make("sgd", model, optimizer)

    # Make a solver
    slv = Solver(env, sgd)

    slv.train_dataset_with_validation(args.epochs)
    #hyper_params = {"learning_rate": args.lr, "epochs":args.epochs,
    #"batch_size":args.batch_size}
    #experiment.log_multiple_params(hyper_params)
    #experiment = Experiment(api_key = "5xNPTUDWzZVquzn8R9oEFkUaa",
    #                        project_name="mnist_examples", workspace="aromorin")
    #experiment.log_metric("Validation accuracy (%)", test_acc)


if __name__ == '__main__':
    main()































#
