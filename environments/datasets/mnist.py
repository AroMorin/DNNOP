"""Class of MNIST dataset"""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset
import matplotlib.pyplot as plt

class MNIST(Dataset):
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, batch_size, data_path, precision=torch.float):
        super().__init__() # Initialize base class
        self.batch_size = batch_size
        self.data_path = data_path
        self.device = torch.device("cuda")
        self.precision = precision

    def load_dataset(self):
        """This dataset is organized as such: it is a list of batches. Each
        batch contains a 2-D Tensor. The first dimension in the Tensor contains
        N Float Tensors (representing images), where N is the batch size. The second
        dimension contains N Long Tensors, corresponding to N labels.
        """
        self.set_transformations()
        self.train_dataset = datasets.MNIST(self.data_path, train=True,
                                            download=True,
                                            transform=self.transforms)
        #self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=60000)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        self.train_set =  list(self.train_loader)
        self.format_set()
        exit()
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        self.test_set = list(self.test_loader)
        self.test_set = self.test_set.to(self.precision)

    def set_transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def format_set(self):
        self.train_set = torch.Tensor (self.train_set)
        self.train_set = self.train_set[0][0].to(self.precision)

    def show_image(self):
        plt.figure()
        train = True
        batch = 0
        mode = 0 #0 for images, 1 for labels
        idx = 0
        if train:
            image = torch.squeeze(self.train_set[batch][image][idx])
        else:
            image = torch.squeeze(self.test_set[batch][image][idx])
        plt.imshow(image)
        plt.show()

    def set_precision(self, precision=torch.float):
            self.precision = precision





#
