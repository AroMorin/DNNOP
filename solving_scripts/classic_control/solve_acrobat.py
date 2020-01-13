"""solve cartpole"""

from __future__ import print_function
import sys, os

from comet_ml import Experiment

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
    module = "Acrobot-v1"  # Max score is -60.0

    # Parameter and Object declarations
    env_params = {
                    "score type": "score",  # Function evaluation
                    "render": True,
                    "Discrete": True,
                    "module name": module
                    }
    env = env_factory.make_env("openai", "control", env_params)

    #print(env.obs_space)
    #print(env.obs_space.high)
    #print(env.obs_space.low)
    #print(env.action_space)
    #print(env.action_space.high)
    #print(env.action_space.low)
    #exit()
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "He",
                    "grad": False,
                    "in features": 6,
                    "number of outputs": 3
                    }
    model = model_factory.make_model("Roboschool Simple FC", model_params)

    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": 0.1,
                    "tolerance": 0.01,
                    "max steps": 64,
                    "memory size": 10
                    }
    alg = algorithm_factory.make_alg("random search", model, alg_params)

    experiment = Experiment(api_key="5xNPTUDWzZVquzn8R9oEFkUaa",
                        project_name="jeff-trinkle", workspace="aromorin")
    experiment.set_name("Acrobot RS")
    hyper_params = {"Algorithm": "RS",
                    "Parameterization": 1000000,
                    "Decay Factor": 0.3,
                    "Directions": 25000,
                    "Search Radius": 0.5
                    }
    experiment.log_parameters(hyper_params)

    slv_params = {
                    "environment": env,
                    "algorithm": alg,
                    "logger": experiment
                    }
    slv = solver_factory.make_slv("RL", slv_params)

    # Use solver to solve the problem
    slv.solve(iterations=500, ep_len=500)
    #slv.solve_online(iterations=1000)
    #slv.solve_online_render(iterations=1000, ep_len=15000)
    #slv.solve_aggregator(iterations=500, reps=10, ep_len=150)
    #slv.solve_averager(iterations=250, reps=10, ep_len=500)
    slv.demonstrate_env(episodes=3, ep_len=500)
    slv.save_elite_weights(alg.model, path='', name='acrobot')

if __name__ == '__main__':
    main()




#
