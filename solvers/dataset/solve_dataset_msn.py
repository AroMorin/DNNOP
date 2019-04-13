"""This script attempts to solve the classification problem of the MNIST
dataset. This specific script uses the MSN algorithm to solve the problem.
Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('../..'))

import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
from backend.solver import Solver

import torch

def main():
    precision = torch.half
    pool_size = 50
    # Make an MNIST Dataset environment
    env_params = {
                    "data path": "~/Documents/ahmed/mnist_data",
                    "precision": precision,
                    "score type": "accuracy",
                    "loss type": "NLL loss",
                    "batch size": 2000  # Entire set
                    }
    env = env_factory.make_env("dataset", "mnist", env_params)

    # Make a pool
    model_params = {
                    "pool size": pool_size,
                    "precision": precision,
                    "weight initialization scheme": "Default"  # Xavier Normal
                    }
    pool = model_factory.make_pool("MNIST CNN MSN", model_params)

    # Make an algorithm --algorithm takes control of the pool--
    alg_params = {
                    "pool size": pool_size,
                    "number of anchors": 3,
                    "number of probes per anchor": 13,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.1,  # Percentage
                    "minimum distance": 1000,
                    "patience": 20,
                    "tolerance": 0.01,
                    "learning rate": 0.02,
                    "lambda": 5,
                    "step size": 0.02
                    }
    alg = algorithm_factory.make_alg("MSN", pool, alg_params)

    # Make a solver
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.train_dataset_with_validation(iterations=500)
    #slv.repeated_batch_train_dataset_with_validation(args.iterations)

if __name__ == '__main__':
    main()








#
