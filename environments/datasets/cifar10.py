"""Class of CIFAR10 dataset"""

from .dataset import Dataset

class CIFAR10(Dataset):
    def __init__(self, batch_size, data_path, precision):
        super().__init__(batch_size, data_path, precision)
