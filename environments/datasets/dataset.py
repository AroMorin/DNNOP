"""Factory class for a Dataset"""
from .mnist.mnist import MNIST
from .cifar10.cifar10 import CIFAR10

class Dataset:
    def factory(name, batch_size, data_path):
        if name == 'mnist':
            return MNIST(batch_size, data_path)
        elif name == 'cifar10':
            return CIFAR10(batch_size, data_path)
