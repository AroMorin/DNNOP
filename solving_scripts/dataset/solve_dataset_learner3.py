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
    #data_path = "C:/Users/aaa2cn/Documents/fashion_mnist_data"
    data_path = "~/Documents/ahmed/fashion_mnist_data"
    # Make an MNIST Dataset environment
    env_params = {
                    "data path": data_path,
                    "precision": precision,
                    "score type": "loss",
                    "loss type": "NLL loss",
                    "batch size": 5000  # Entire set
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
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "tolerance": 0.01,
                    "learning rate": 0.05,
                    "lambda": 5,
                    "alpha": 0.05,
                    "beta": 0.29,
                    "minimum entropy": -0.1,
                    "max steps": 50
                    }
    alg = algorithm_factory.make_alg("learner3", model, alg_params)

    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("dataset", slv_params)

    # Use solver to solve the problem
    slv.train_dataset_with_validation(iterations=15000)
    #slv.batch_train_dataset_with_validation(iterations=2500)
    #slv.repeated_batch_train_dataset_with_validation(iterations=5, reps=500)

if __name__ == '__main__':
    main()








#
