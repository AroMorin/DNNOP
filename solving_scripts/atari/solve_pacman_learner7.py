"""solve atari"""
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
    # Variable definition
    precision = torch.float
    game = "Pong-ram-v0"

    # Parameter and Object declarations
    env_params = {
                    "score type": "score",  # Function evaluation
                    "render": False,
                    "RAM": True,
                    "game name": game,
                    }
    env = env_factory.make_env("openai", "atari", env_params)

    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Spiking",
                    "number of outputs": env.action_space.n,
                    "w": 210,
                    "h": 160,
                    "in features": 128,
                    "in channels": 3
                    }
    model = model_factory.make_model("DQN RAM model", model_params)

    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "tolerance": 0.01,
                    "minimum entropy": 0.1,
                    "max steps": 50,
                    "memory size": 10
                    }
    alg = algorithm_factory.make_alg("learner7", model, alg_params)


    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("RL", slv_params)

    # Use solver to solve the problem
    slv.solve(iterations=200)
    slv.demonstrate_env()

if __name__ == '__main__':
    main()




#
