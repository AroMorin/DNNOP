"""This script allows making objects of different dataset classes"""

from .mnist import MNIST
from .cifar10 import CIFAR10

def make(name, batch_size, data_path):
    """Class factory method. This method takes the name of the desired
    dataset and returns an object of said class.
    """
    if name == 'mnist':
        return MNIST(batch_size, data_path)
    elif name == 'cifar10':
        return CIFAR10(batch_size, data_path)
