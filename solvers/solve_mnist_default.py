"""This script attempts to solve the classification problem of the MNIST
dataset. Comet ML is used to automatically upload and document the results.
"""
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from backend.models.cnn_mnist import Net
import environments.datasets as datasets
import torch.nn.functional as F
#from comet_ml import Experiment

import argparse
import torch
import torch.optim as optim

def train(model, dataset, optimizer, epoch):
    model.train() #Sets behavior to "training" mode
    for i in range(len(dataset.train_data)):
        optimizer.zero_grad()
        output = model(dataset.train_data[i])
        loss = F.nll_loss(output, dataset.train_labels[i])
        loss.backward()
        optimizer.step()
    print('Train Loss: %f' %loss.item())

def test(model, dataset):
    data = dataset.test_data
    labels = dataset.test_labels
    nb_images = len(dataset.test_data)
    model.eval() #Sets behavior to "training" mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, labels, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= nb_images
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, nb_images, 100.*correct/nb_images))
    return 100.*correct/nb_images

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
    device = torch.device("cuda")
    model = Net().half().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    data_path = "C:/Users/aaa2cn/Documents/mnist_data"

    #make an object of the dataset class
    dataset = datasets.make("mnist", args.batch_size, data_path)
    dataset.load_dataset()
    print(len(dataset.train_data))

    for epoch in range(1, args.epochs+1):
        train(model, dataset, optimizer, epoch)
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
