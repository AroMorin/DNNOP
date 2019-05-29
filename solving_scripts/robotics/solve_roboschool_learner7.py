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
    #game = "Pong-v0"
    #game = "Pong-ram-v0"
    module = "RoboschoolReacher-v1"

    # Parameter and Object declarations
    env_params = {
                    "score type": "score",  # Function evaluation
                    "render": False,
                    "module name": module
                    }
    env = env_factory.make_env("openai", "roboschool", env_params)

    #print(env.obs_space)
    #print(env.action_space)
    #exit()
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Sparse",
                    "grad": False,
                    "in features": 9,
                    "number of outputs": 2
                    }
    model = model_factory.make_model("Roboschool FC", model_params)

    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.1,
                    "tolerance": 0.01,
                    "learning rate": 0.2,
                    "lambda": 5,
                    "alpha": 0.0005,
                    "beta": 0.29,
                    "max steps": 30,
                    "memory size": 10
                    }
    alg = algorithm_factory.make_alg("learner7", model, alg_params)


    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("RL", slv_params)

    # Use solver to solve the problem
    slv.solve(iterations=5000)
    slv.demonstrate_env()

if __name__ == '__main__':
    main()




#
