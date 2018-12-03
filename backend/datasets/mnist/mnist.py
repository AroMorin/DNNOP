"""Class of MNIST dataset"""
import torch
from torchvision import datasets, transforms

class MNIST():
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, batch_size, data_path):
        self.batch_size = batch_size
        self.data_path = data_path
        self.train_dataset = ''
        self.train_loader = ''
        self.test_dataset = ''
        self.test_loader = ''
        self.transforms = ''
        self.dataset = ''
        self.device = torch.device("cuda")

    def init_dataset(self):
        self.train_dataset = datasets.MNIST(self.data_path, train=True,
                                            download=True,
                                            transform=self.transforms)
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        self.train_dataset.to(self.device)
        self.test_dataset.to(self.device)

    def load_dataset(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)

    def transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
