"""This script solves a global optimization function by finding the location
of its global optimum.
Currently, only MSN algorithm is avaiable to solve this problem.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('../../..'))
import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
from backend.solver import Solver

import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='NAO Solver')
    parser.add_argument('--pool_size', type=int, default=50, metavar='N',
                        help='number of samples in the pool (def: 50)')
    parser.add_argument('--iterations', type=int, default=500, metavar='N',
                        help='maximum number of optimization steps (def: 500)')
    args = parser.parse_args()

    # Make an environment object
    env_params = {
                "data_path": "C:/Users/aaa2cn/Documents/nao_data/",
                "ip": "localhost",
                "port": 46251
                }
    env = env_factory.make_env("nao", "pose assumption", env_params)
    env.say("Sup my man!")
    exit()
    sensors = env.get_joints()
    print(sensors)

    # Make a pool object
    model_params = {
                    "precision": torch.float,
                    "weight initialization scheme": "Normal"
                    }
    pool = model_factory.make_pool("NAO FC model", args.pool_size, model_params)

    # Make an algorithm object
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
    alg = algorithm_factory.make_alg("MSN", pool, alg_params)

    # Make a solver object
    slv = Solver(env, alg)

    # Use solver to solve the problem
    slv.solve(args.iterations)

if __name__ == '__main__':
    main()










#
