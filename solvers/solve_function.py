"""solve a function"""
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
    parser = argparse.ArgumentParser(description='Func Solver')
    parser.add_argument('--pool_size', type=int, default=50, metavar='N',
                        help='number of samples in the pool (def: 50)')
    parser.add_argument('--nb_anchors', type=int, default=5, metavar='N',
                        help='number of anchors (def: 5)')
    parser.add_argument('--nb_probes', type=int, default=8, metavar='N',
                        help='number of probes per anchor (def: 8)')
    parser.add_argument('--iterations', type=int, default=500, metavar='N',
                        help='maximum number of optimization steps (def: 500)')
    args = parser.parse_args()

    data_path = "C:/Users/aaa2cn/Documents/function_data/rastrigin"
    precision = torch.half # Set precision

    # Make an MNIST Dataset environment
    env = environments.make_env("function",
                                "rastrigin",
                                data_path = data_path,
                                plot = False,
                                precision = precision
                                )
    # Make a pool
    pool = model_factory.make_pool("Function FC model", args.pool_size, precision)

    # Make an algorithm --algorithm takes control of the pool--
    hyper_params = {
                    "pool size": args.pool_size,
                    "number of anchors": args.nb_anchors,
                    "number of probes per anchor": args.nb_probes,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": -5,  # Percentage
                    "minimum distance": 300
                    }
    alg = algorithm_factory.make_alg("MSN", pool, hyper_params)

    # Make a solver
    slv = Solver(env, alg)

    # Use solver to solve the problem
    #slv.train_dataset_with_validation(args.iterations)
    slv.solve(args.iterations)

if __name__ == '__main__':
    main()
















#
