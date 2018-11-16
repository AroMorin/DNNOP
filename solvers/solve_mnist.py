"""This script attempts to solve the classification problem of the MNIST
dataset
"""
from comet_ml import Experiment

experiment = Experiment(api_key = "5xNPTUDWzZVquzn8R9oEFkUaa",
                        project_name="general", workspace="aromorin")

hyper_params = {"learning_rate": 0.5, "steps":100000, "batch_size":50}
experiment.log_multiple_params(hyper_params)

train_acc = 100
experiment.log_metric("training accuracy (%)", train_acc)
