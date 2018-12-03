"""Factory class for a Dataset"""

class Dataset:
    def factory(name, batch_size, data_path):
        if name == 'mnist':
            return MNIST(batch_size, data_path)
        elif name == 'cifar10':
            return CIFAR10(batch_size, data_path)
