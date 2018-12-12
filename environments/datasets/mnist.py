"""Class of MNIST dataset"""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset

class MNIST(Dataset):
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, batch_size, data_path):
        super().__init__() # Initialize base class
        self.batch_size = batch_size
        self.data_path = data_path
        self.device = torch.device("cuda")

    def init_dataset(self):
        self.transformations()
        self.train_dataset = datasets.MNIST(self.data_path, train=True,
                                            download=True,
                                            transform=self.transforms)
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)

    def transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def get_train_data(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        self.batches = list(enumerate(self.train_loader))
        print(len(self.batches[:]))
        #self.train_data, self.train_labels = self.batches
        exit()
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True)
        return self.train_data

    def get_train_labels(self):
        pass

    def get_test_data(self):
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
        batch_size=self.batch_size,
        shuffle=True)

    def get_test_labels(self):
        pass

    def load_dataset(self):
        self.batches = list(enumerate(self.train_loader))
        self.train_data.to(self.device)
        self.test_loader.to(self.device)
