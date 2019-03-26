"""This script attempts to solve the classification problem of the MNIST
dataset. This specific script uses the MSN algorithm to solve the problem.
Comet ML is used to automatically upload and document the results.
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

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

def main():
    # Assumes CUDA is available
    blockPrint()
    path = str(sys.argv[1])


    # Define parameters
    env_params = {
                    "path": path,
                    "precision": torch.float,
                    "score type": "score",
                    "inference": True
                    }
    env = env_factory.make_env("task", "object detection", env_params)

    model_params = {
                    "precision": torch.float,
                    "weight initialization scheme": "Default",  # Xavier Normal,
                    "pre-trained": True,
                    "path": "~/find_phone/model_elite.pth"
                    }
    model = model_factory.make_model("OD CNN MSN", model_params)

    alg_params = {
                    "number of anchors": 3,
                    "number of probes per anchor": 13,
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "minimum entropy": -1,  # Percentage
                    "minimum distance": 200,
                    "patience": 20,
                    "tolerance": 0.01,
                    "learning rate": 0.01,
                    "lambda": 5
                    }

    image = env.get_image(path)
    inference = model(image)

    x = round(inference[0][0].item(), 4)
    y = round(inference[0][1].item(), 4)

    enablePrint()
    print(str(x)+" "+str(y))

if __name__ == '__main__':
    main()





#
