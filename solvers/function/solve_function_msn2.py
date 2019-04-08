"""This script solves a global optimization function by finding the location
of its global optimum.
Currently, only MSN algorithm is avaiable to solve this problem.
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
    precision = torch.float
    # Make a function environment
    env_params = {
                    "data path": "~/Documents/ahmed/function_data/rastrigin/",
                    "precision": precision,
                    "plot": False,
                    "score type": "score"  # Function evaluation
                    }
    env = env_factory.make_env("function", "rastrigin", env_params)

    # Make a pool
    model_params = {
                    "pool size": 50,
                    "precision": precision,
                    "weight initialization scheme": "Identical"
                    }
    pool = model_factory.make_pool("Function FC model", model_params)

    # Make an algorithm --algorithm needs to take charge of the pool--
    alg_params = {
                    "pool size": 50,
                    "number of anchors": 5,
                    "number of probes per anchor": 8,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": -3,  # Percentage
                    "minimum distance": 400,
                    "patience": 27,
                    "tolerance": 0.12
                    }
    alg = algorithm_factory.make_alg("MSN2", pool, alg_params)

    # Make a solver using the environment and algorithm objects
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.solve_and_plot(iterations=500)

if __name__ == '__main__':
    main()




#
