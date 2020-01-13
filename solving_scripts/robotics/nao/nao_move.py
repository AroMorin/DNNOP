"""This script attempts to solve the robot posture assumption problem."""

from __future__ import print_function
import sys, os

from comet_ml import Experiment

# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('../../..'))
import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
import backend.solvers as solver_factory

import argparse
import torch

def main():
    precision = torch.float

    # Parameter and Object declarations
    env_params = {
                    "data path": "C:/Users/aaa2cn/Documents/nao_data/",
                    "ip": "localhost",
                    "port": 52232,
                    "score type": "score"  # Aggregate error in pose
                    }
    env = env_factory.make_env("nao", "pose assumption", env_params)

    # Make a pool object
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Sparse",
                    "grad": False
                    }
    model = model_factory.make_model("NAO FC model", model_params)

    # Make an algorithm object
    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.1,
                    "tolerance": 0.1,
                    "max steps": 64,
                    "memory size": 10
                    }
    alg = algorithm_factory.make_alg("local search", model, alg_params)

    experiment = Experiment(api_key="5xNPTUDWzZVquzn8R9oEFkUaa",
                        project_name="nao", workspace="aromorin")
    experiment.set_name("Pose Assumption virtual")
    hyper_params = {"Algorithm": "LS",
                    "Parameterization": 35000,
                    "Decay Factor": 0.01,
                    "Directions": 10,
                    "Search Radius": 0.1
                    }
    experiment.log_parameters(hyper_params)

    slv_params = {
                    "environment": env,
                    "algorithm": alg,
                    "logger": experiment
                    }
    slv = solver_factory.make_slv("robot", slv_params)
    slv.solve(iterations=5000)

    slv.save_elite_weights(path='', name='pose_assump_virtual')

    # Recreate the target pose
    alg.eval()
    pred = alg.model(env.observation)
    angles = [p.item() for p in pred]
    print("These are the angles: ")
    print(angles)
    env.set_joints(angles)
    env.say("Is this the pose you set for me?")
    env.rest()


if __name__ == '__main__':
    main()










#
