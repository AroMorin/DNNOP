"""Class of MNIST dataset."""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset
import matplotlib.pyplot as plt

class FashionMNIST(Dataset):
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, env_params):
        # Initialize base class
        super(MNIST, self).__init__(env_params)
        env_params = self.ingest_params_lvl2(env_params)
        self.train_size = 60000 # Size of the training set
        self.test_size = 10000
        self.batch_size = (env_params["batch size"])
        self.load_dataset()

    def ingest_params_lvl2(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "batch size": 60000  # Entire dataset
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def load_dataset(self):
        """This dataset is organized as such: it is a list of batches. Each
        batch contains a 2-D Tensor. The first dimension in the Tensor contains
        N Float Tensors (representing images), where N is the batch size. The second
        dimension contains N Long Tensors, corresponding to N labels.
        """
        self.set_transformations()
        # Initialize and load training set
        self.train_dataset = datasets.FashionMNIST(self.data_path, train=True,
                                                    download=True,
                                                    transform=self.transforms)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.train_size,
                                                    shuffle=True,
                                                    pin_memory = True,
                                                    num_workers = 8)
        # Initialize and load validation set
        self.test_dataset = datasets.FashionMNIST(self.data_path, train=False,
                                                    download=True,
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
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                (0.5, 0.5, 0.5))
                                            ])

    def format_data(self):
        """Apply the desired precision, split into batches and perform any other
        similar operations on the dataset.
        """
        train_set =  list(self.train_loader)
        test_set = list(self.test_loader)

        # batch 0: all images, mode 0: data
        x = train_set[0][0].to(dtype=self.precision, device=self.device)
        self.test_data = test_set[0][0].to(dtype=self.precision, device=self.device)

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
        """Checks whether we've reached reset condition or not."""
        # If reached end of batches, reset
        return self.current_batch_idx>=self.nb_batches

    def reset(self):
        """Reset class state."""
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
