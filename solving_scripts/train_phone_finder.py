"""This script attempts to solve the classification problem of the MNIST
dataset. This specific script uses the MSN algorithm to solve the problem.
Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('..'))

import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
from backend.solver import Solver

import argparse
import torch

def main():
    # Assumes CUDA is available
    #def_path = "C:/Users/aaa2cn/Documents/phone_data/find_phone/"
    path = str(sys.argv[1])

    # Define parameters
    env_params = {
                    "path": path,
                    "precision": torch.float,
                    "score type": "score"
                    }
    env = env_factory.make_env("task", "object detection", env_params)

    model_params = {
                    "pool size": 50,
                    "precision": torch.float,
                    "weight initialization scheme": "Default"  # Xavier Normal
                    }
    pool = model_factory.make_pool("OD CNN MSN", model_params)

    alg_params = {
                    "number of anchors": 3,
                    "number of probes per anchor": 13,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": -0.1,  # Percentage
                    "minimum distance": 1000,
                    "patience": 20,
                    "tolerance": 0.01,
                    "learning rate": 0.02,
                    "lambda": 5,
                    "step size": 0.02
                    }
    alg = algorithm_factory.make_alg("MSN", pool, alg_params)

    slv = Solver(env, alg)
    # Use solver to solve the environment using the given algorithm
    slv.solve(iterations=1000)
    alg.save_weights(path)

if __name__ == '__main__':
    main()





#