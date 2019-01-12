"""This script attempts to solve the classification problem of the MNIST
dataset. Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('..'))
import environments
import backend.models as model_factory
import backend.algorithms as algorithm_factory
from solver import Solver

import argparse
import torch

def main():
    # Assumes CUDA is available
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--pool_size', type=int, default=50, metavar='N',
                        help='number of samples in the pool (default: 50)')
    parser.add_argument('--nb_anchors', type=int, default=4, metavar='N',
                        help='number of anchors (default: 4)')
    parser.add_argument('--nb_probes', type=int, default=8, metavar='N',
                        help='number of probes per anchor (default: 8)')
    parser.add_argument('--iterations_limit', type=int, default=500, metavar='N',
                        help='maximum allowable number of optimization steps (default: 500)')
    args = parser.parse_args()

    # Set desired precision
    precision = torch.half
    training_size = 60000
    hyper_params = {"pool size": pool_size,
                    "number of anchors": nb_anchors,
                    "number of probes per anchor": nb_probes}

    # Make an MNIST Dataset environment
    data_path = "C:/Users/aaa2cn/Documents/mnist_data"
    env = environments.make("dataset", "mnist", training_size, data_path, precision)

    # Make a pool
    pool = model_factory.make_pool("MNIST CNN", pool_size, precision)

    # Make an algorithm --algorithm takes control of the pool--
    alg = algorithm_factory.make("MSN", pool, hyper_params)

    # Make a solver
    slv = Solver(env, alg)

    slv.train_dataset_with_validation(args.iterations)


if __name__ == '__main__':
    main()































#