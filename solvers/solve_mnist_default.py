"""This script attempts to solve the classification problem of the MNIST
dataset. Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import environments
import backend.models as model_factory
import backend.algorithms as algorithms_factory
from solver import Solver
#from comet_ml import Experiment

import argparse
import torch

def main():
    # Assumes CUDA is available
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    precision = torch.half # Set precision

    # Make an MNIST Dataset environment
    data_path = "C:/Users/aaa2cn/Documents/mnist_data"
    env = environments.make_env("dataset", "mnist", batch_size=args.batch_size,
                                    data_path=data_path, precision=precision)

    # Make a model
    model = model_factory.make_model("MNIST CNN", precision)

    # Make an algorithm --algorithm takes control of the model--
    hyper_params = {
                    "learning rate": 0.01,
                    }
    alg = algorithms_factory.make_alg("sgd", model, hyper_params)

    # Make a solver
    slv = Solver(env, alg)

    slv.batch_train_dataset_with_validation(args.epochs)
    #hyper_params = {"learning_rate": args.lr, "epochs":args.epochs,
    #"batch_size":args.batch_size}
    #experiment.log_multiple_params(hyper_params)
    #experiment = Experiment(api_key = "5xNPTUDWzZVquzn8R9oEFkUaa",
    #                        project_name="mnist_examples", workspace="aromorin")
    #experiment.log_metric("Validation accuracy (%)", test_acc)


if __name__ == '__main__':
    main()































#
