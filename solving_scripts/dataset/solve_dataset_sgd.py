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
import backend.solvers as solver_factory

import torch

def main():
    precision = torch.half
    # Make an MNIST Dataset environment
    env_params = {
                    "data path": "~/Documents/ahmed/fashion_mnist_data",
                    "precision": precision,
                    "score type": "loss",
                    "loss type": "NLL loss",
                    "batch size": 10000  # Entire set
                    }
    env = env_factory.make_env("dataset", "fashion mnist", env_params)

    # Make a pool
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Default"  # Xavier Normal
                    }
    model = model_factory.make_model("FashionMNIST CNN", model_params)

    # Make an algorithm --algorithm takes control of the pool--
    alg_params = {
                    "learning rate": 0.01,
                    }
    alg = algorithm_factory.make_alg("sgd", model, alg_params)

    # Make a solver
    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("dataset", slv_params)

    # Use solver to solve the problem
    slv.train_dataset_with_validation(iterations=2500)
    #slv.batch_train_dataset_with_validation(iterations=62)

if __name__ == '__main__':
    main()








#
