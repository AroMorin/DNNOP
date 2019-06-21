"""This script attempts to solve the classification problem of the MNIST
dataset. This specific script uses the MSN algorithm to solve the problem.
Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
# Append SYSPATH in order to access different modules of the library
sys.path.insert(0, os.path.abspath('../..'))

import environments as env_factory
import backend.models as model_factory
import backend.algorithms as algorithm_factory
import backend.solvers as solver_factory
import torchvision.models as models

import torch

def main():
    precision = torch.float
    #data_path = "C:/Users/aaa2cn/Documents/fashion_mnist_data"
    #data_path = "~/Documents/ahmed/fashion_mnist_data"
    data_path = "~/Documents/ahmed/cifar10_data"
    # Make an MNIST Dataset environment
    env_params = {
                    "data path": data_path,
                    "precision": precision,
                    "score type": "accuracy",
                    "loss type": "NLL loss",
                    "normalize": True,
                    "batch size": 1000  # Entire set
                    }
    env = env_factory.make_env("dataset", "cifar10", env_params)

    # Make a pool
    model_params = {
                    "precision": precision,
                    "weight initialization scheme": "Normal"
                    }
    #model = model_factory.make_model("CIFAR10 CNN", model_params)
    model = vgg16 = models.resnet18().cuda()

    # Make an algorithm --algorithm takes control of the pool--
    alg_params = {
                    "target": env.target,
                    "minimization mode": env.minimize,
                    "tolerance": 0.01,
                    "minimum entropy": 0.1,
                    "max steps": 100,
                    "memory size": 101
                    }

    alg = algorithm_factory.make_alg("learner", model, alg_params)

    slv_params = {
                    "environment": env,
                    "algorithm": alg
                    }
    slv = solver_factory.make_slv("dataset", slv_params)

    # Use solver to solve the problem
    slv.train_dataset_with_validation(iterations=3000)
    #slv.determined_batch_train_with_validation(iterations=5)
    #slv.repeated_batch_train_dataset_with_validation(iterations=3, reps=3000)

if __name__ == '__main__':
    main()








#
