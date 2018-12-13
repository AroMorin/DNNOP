"""Class of MNIST dataset"""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset
import matplotlib.pyplot as plt

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
        """This dataset is organized as such: it is a list of batches. Each
        batch contains a 2-D Tensor. The first dimension in the Tensor contains
        N Float Tensors (representing images), where N is the batch size. The second
        dimension contains N Long Tensors, corresponding to N labels.
        """
        self.implement_transformations()
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
        self.show_image()
        exit()
        #Use inverse zip expression to unpack the (batch_idx, data) tuples
        _, self.train_tuples = zip(*self.train_iterable)
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        self.test_tuples = list(enumerate(self.test_loader))
        #Use inverse zip expression to unpack the (batch_idx, data) tuples
        _, self.test_set = zip(*self.test_tuples)

    def implement_transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def show_image(self):
        plt.figure()
        idx = 0
        image = torch.squeeze(self.train_set[0][0][idx])
        print (image)
        plt.imshow(image)
        plt.show()
        image = image.to(torch.int8)
        print(image.dtype)
        image = image.to(torch.float)
        print (image)
        plt.imshow(image)
        plt.show()

    def get_train_batches(self):
        print (len(self.train_data))
        exit()
        # Convert tuple into list, then convert into a Torch tensor
        self.train_data = torch.tensor((self.train_data[:][0]))
        print(self.train_data)
        exit()
        self.train_labels.to(self.device)
        #self.train_data, self.train_labels = self.batches
        return self.train_data, self.train_labels

    def get_test_batchess(self):
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset)
