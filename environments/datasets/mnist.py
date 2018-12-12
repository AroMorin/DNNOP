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
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=60000)
        #self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
        #batch_size=self.batch_size,
        #shuffle=True)
        self.train_tuples = enumerate(self.train_loader)
        #Use inverse zip expression to unpack the (batch_idx, data) tuples
        #_, self.train_set = zip(*self.train_tuples)
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        self.test_tuples = list(enumerate(self.test_loader))
        #Use inverse zip expression to unpack the (batch_idx, data) tuples
        _, self.test_set = zip(*self.test_tuples)

    def transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def get_train_set(self):
        print (len(self.train_data))
        exit()
        # Convert tuple into list, then convert into a Torch tensor
        self.train_data = torch.tensor((self.train_data[:][0]))
        print(self.train_data)
        exit()
        self.train_labels.to(self.device)
        #self.train_data, self.train_labels = self.batches
        return self.train_data, self.train_labels

    def get_test_set(self):
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset)
