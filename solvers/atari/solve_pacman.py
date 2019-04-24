"""solve atari"""
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
    game = "MsPacman"
    env_params = {
                    "score type": "score",  # Function evaluation
                    "render": True,
                    "RAM": True
                    }
    env = env_factory.make_env("openai", game, env_params)

    # Make a model
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Default",
                    "number of outputs": 8,
                    "w": 210,
                    "h": 160,
                    "in features": 128,
                    "in channels": 3
                    }
    model = model_factory.make_model("DQN RAM model", model_params)

    # Make an algorithm --algorithm needs to take charge of the pool--
    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.1,
                    "patience": 3000,
                    "tolerance": 0.01,
                    "learning rate": 0.05,
                    "lambda": 5,
                    "alpha": 0.05,
                    "beta": 0.29,
                    "step size": 0.2
                    }
    alg = algorithm_factory.make_alg("learner2", model, alg_params)

    # Make a solver using the environment and algorithm objects
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.solve_env(iterations=100)

if __name__ == '__main__':
    main()




#
