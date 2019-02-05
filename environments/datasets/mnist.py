"""Class of MNIST dataset"""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset
import matplotlib.pyplot as plt

class MNIST(Dataset):
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, data_path, batch_size, precision):
        # Initialize base class
        super().__init__(data_path, precision)
        self.train_size = 60000 # Size of the training set
        self.test_size = 10000
        self.set_batch_size(batch_size)
        self.load_dataset()

    def load_dataset(self):
        """This dataset is organized as such: it is a list of batches. Each
        batch contains a 2-D Tensor. The first dimension in the Tensor contains
        N Float Tensors (representing images), where N is the batch size. The second
        dimension contains N Long Tensors, corresponding to N labels.
        """
        self.set_transformations()
        # Initialize and load training set
        self.train_dataset = datasets.MNIST(self.data_path, train=True,
                                            download=True,
                                            transform=self.transforms)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.train_size,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        # Initialize and load validation set
        self.test_dataset = datasets.MNIST(self.data_path, train=False,
                                            transform=self.transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.test_size,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        # Format sets
        self.format_data()


    def set_transformations(self):
        """Set the desired transformations on the dataset."""
        print ("Applying dataset transformations")
        self.transforms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

    def format_data(self):
        """Apply the desired precision, split into batches and perform any other
        similar operations on the dataset.
        """
        train_set =  list(self.train_loader)
        test_set = list(self.test_loader)

        # batch 0: all images, mode 0: data
        x = train_set[0][0].cuda().to(self.precision)
        self.test_data = test_set[0][0].cuda().to(self.precision)

        # batch 0: all images, mode 1: labels
        y = train_set[0][1].cuda()
        self.test_labels = test_set[0][1].cuda()

        self.train_data = torch.split(x, self.batch_size)
        self.train_labels = torch.split(y, self.batch_size)
        self.nb_batches = len(self.train_data)
        print ("Number of Batches: %d" %self.nb_batches)

    def step(self):
        """Loads a batch of images and labels.
        This method can be further customized to randomize the batch
        contents.
        """
        self.observation = self.train_data[self.current_batch_idx]
        self.labels = self.train_labels[self.current_batch_idx]
        self.current_batch_idx += 1
        if self.check_reset():
            self.reset()

    def check_reset(self):
        # If reached end of batches, reset
        return self.current_batch_idx>=self.nb_batches

    def reset(self):
            self.current_batch_idx = 0

    def show_image(self):
        """Method to show the user an image from the dataset."""
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
        """In case the user wanted to change the precision after loading the
        dataset."""
        self.precision = precision





#
