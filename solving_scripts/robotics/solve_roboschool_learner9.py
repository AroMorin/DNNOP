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
    #module = "RoboschoolReacher-v1"
    #module = "RoboschoolAnt-v1"
    module = "RoboschoolInvertedPendulum-v1"

    # Parameter and Object declarations
    env_params = {
                    "score type": "score",  # Function evaluation
                    "render": False,
                    "module name": module
                    }
    env = env_factory.make_env("openai", "roboschool", env_params)

    #print(env.action_space.low)
    #print(env.action_space.high)
    #print(env.obs_space)
    #print(env.action_space)
    #exit()

    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Constant",
                    "grad": False,
                    "in features": 5,
                    "number of outputs": 1,
                    "min action 1": -1.,
                    "max action 1": 1.,
                    "noise limit": 0.05
                    }
    model = model_factory.make_model("Roboschool FC", model_params)

    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.5,
                    "tolerance": 0.01,
                    "max steps": 256,
                    "memory size": 2
                    }
    alg = algorithm_factory.make_alg("learner9", model, alg_params)


    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("RL", slv_params)

    # Use solver to solve the problem
    slv.solve_online(iterations=1000)
    #slv.solve_averager(iterations=10000, reps=3)
    slv.demonstrate_env(episodes=5)
    slv.save_elite_weights(alg.model, path='')

if __name__ == '__main__':
    main()




#
