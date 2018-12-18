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
        self.precision = torch.half

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
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        # Load entire set as one batch, format it, then split it into batches
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=60000,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=10000,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        self.format_data()
        print(self.train_data[0].dtype)
        exit()

    def set_transformations(self):
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def format_data(self):
        train_set =  list(self.train_loader)
        test_set = list(self.test_loader)

        # batch 0, ie. all images, and mode 0 (ie. data not labels)
        train_data = train_set[0][0].to(self.precision)
        test_data = test_set[0][0]

        train_labels = train_set[0][1]
        test_labels = test_set[0][1]

        self.train_data = torch.split(train_data, self.batch_size)
        self.test_data = torch.split(test_data, self.batch_size)

        self.train_labels = torch.split(train_labels, self.batch_size)
        self.test_labels = torch.split(test_labels, self.batch_size)

    def show_image(self):
        plt.figure()
        train = True
        batch = 0
        mode = 0 #0 for images, 1 for labels
        i = 0 # Image Index
        if train:
            image = torch.squeeze(self.train_set[batch][mode][image])
        else:
            image = torch.squeeze(self.test_set[batch][mode][i])
        plt.imshow(image)
        plt.show()

    def set_precision(self, precision=torch.float):
            self.precision = precision





#
