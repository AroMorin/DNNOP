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
import backend.solvers as solver_factory

import argparse
import torch

def main():
    precision = torch.float
    # Make a function environment
    function = "rastrigin"
    env_params = {
                    "data path": "function_data/"+function+"/",
                    "precision": precision,
                    "plot": False,
                    "score type": "error",  # Function evaluation
                    "populations": False  # Single-solution optimization
                    }
    env = env_factory.make_env("function", function, env_params)

    # Make a model
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Identical"
                    }
    model = model_factory.make_model("Function FC model", model_params)

    # Make an algorithm --algorithm needs to take charge of the pool--
    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "tolerance": 0.12,
                    "minimum entropy": -0.1,  # Percentage
                    "learning rate": 0.35,
                    "alpha": 0.3,
                    "beta": 0.29,
                    "max steps": 50
                    }
    alg = algorithm_factory.make_alg("learner3", model, alg_params)

    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("function", slv_params)

    # Use solver to solve the problem
    slv.solve(iterations=5000)
    #slv.solve_and_plot(iterations=5000)

if __name__ == '__main__':
    main()




#
