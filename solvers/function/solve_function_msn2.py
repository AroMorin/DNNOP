"""This script solves a global optimization function by finding the location
of its global optimum.
Currently, only MSN algorithm is avaiable to solve this problem.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('../..'))
import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
from backend.solver import Solver

import argparse
import torch

def main():
    precision = torch.float
    # Make a function environment
    function = "rosenbrock"
    env_params = {
                    "data path": "function_data/"+function+"/",
                    "precision": precision,
                    "plot": True,
                    "score type": "error"  # Function evaluation
                    }
    env = env_factory.make_env("function", function, env_params)


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
                    "number of anchors": 4,
                    "number of probes per anchor": 9,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum distance": 430,
                    "patience": 32,
                    "tolerance": 0.12,
                    "learning rate": 0.1,
                    "lambda": 5,
                    "step size": 0.02
                    }
    alg = algorithm_factory.make_alg("MSN2", pool, alg_params)

    # Make a solver using the environment and algorithm objects
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.solve_and_plot(iterations=500)

if __name__ == '__main__':
    main()




#
