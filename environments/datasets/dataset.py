"""Factory class for a Dataset"""
from .mnist import MNIST
from .cifar10 import CIFAR10

class Dataset:
    def __init__(self, name, batch_size, data_path):
        self.batch_size = batch_size
        self.data_path = data_path
        self.name = name

    def make(self):
        if self.name == 'mnist':
            return MNIST(self.batch_size, self.data_path)
        elif self.name == 'cifar10':
            return CIFAR10(self.batch_size, self.data_path)
