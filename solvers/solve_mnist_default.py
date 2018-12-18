"""This script attempts to solve the classification problem of the MNIST
dataset. Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from backend.models.cnn_mnist import Net
import environments
import backend.algorithms as algorithms
import torch.nn.functional as F
#from comet_ml import Experiment

import argparse
import torch
import torch.optim as optim

def main():
    # Assumes CUDA is available
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    model = Net().half().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    data_path = "C:/Users/aaa2cn/Documents/mnist_data"

    #make an object of the dataset class
    env = environments.make("dataset", "mnist", args.batch_size, data_path)

    solver = algorithms.make("sgd")

    for i in steps:
        env.step()
        solver.solve(env, model, optimizer)

    for epoch in range(args.epochs):
        train(model, dataset, optimizer)
        test_acc = test(model, dataset)

    #hyper_params = {"learning_rate": args.lr, "epochs":args.epochs,
    #"batch_size":args.batch_size}
    #experiment.log_multiple_params(hyper_params)
    #experiment = Experiment(api_key = "5xNPTUDWzZVquzn8R9oEFkUaa",
    #                        project_name="mnist_examples", workspace="aromorin")
    #experiment.log_metric("Validation accuracy (%)", test_acc)


if __name__ == '__main__':
    main()































#
