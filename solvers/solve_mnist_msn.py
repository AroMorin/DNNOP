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
    parser.add_argument('--pool_size', type=int, default=15, metavar='N',
                        help='number of samples in the pool (def: 25)')
    parser.add_argument('--nb_anchors', type=int, default=3, metavar='N',
                        help='number of anchors (def: 3)')
    parser.add_argument('--nb_probes', type=int, default=3, metavar='N',
                        help='number of probes per anchor (def: 3)')
    parser.add_argument('--iterations', type=int, default=500, metavar='N',
                        help='maximum number of optimization steps (def: 500)')
    args = parser.parse_args()

    precision = torch.half # Set precision

    # Make an MNIST Dataset environment
    data_path = "C:/Users/aaa2cn/Documents/mnist_data"
    env = environments.make_env("dataset", "mnist", data_path=data_path, precision=precision)

    # Make a pool
    pool = model_factory.make_pool("MNIST CNN", args.pool_size, precision)

    # Make an algorithm --algorithm takes control of the pool--
    hyper_params = {
                    "pool size": args.pool_size,
                    "number of anchors": args.nb_anchors,
                    "number of probes per anchor": args.nb_probes,
                    }
    alg = algorithm_factory.make_alg("MSN", pool, hyper_params)

    # Make a solver
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.train_dataset_with_validation(args.iterations)


if __name__ == '__main__':
    main()































#
