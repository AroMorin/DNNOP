"""Class of Fashion MNIST dataset."""
import torch
from torchvision import datasets, transforms
from .dataset import Dataset

class FashionMNIST(Dataset):
    """This class fetches the MNIST dataset, sends it to CUDA and then
    makes available the PyTorch-supplied loaders for further processing.
    """
    def __init__(self, env_params):
        # Initialize base class
        super(FashionMNIST, self).__init__(env_params)
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
        N Float Tensors (representing images), where N is the batch size. The
        second dimension contains N Long Tensors, corresponding to N labels.
        """
        self.set_transformations()
        # Initialize and load training set
        train_dataset = datasets.FashionMNIST(self.data_path, train=True,
                                                    download=True,
                                                    transform=self.transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.train_size,
                                                    shuffle=True,
                                                    num_workers = 8)
        # Initialize and load validation set
        test_dataset = datasets.FashionMNIST(self.data_path, train=False,
                                                    download=True,
                                                    transform=self.transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=self.test_size,
                                                    shuffle=True,
                                                    num_workers = 8)
        # Format sets
        self.format_data(train_loader, test_loader)


    def set_transformations(self):
        """Set the desired transformations on the dataset."""
        print ("Applying dataset transformations")
        if self.normalize:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,),
                                                                     (0.5,))
                                                  ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])


    def format_data(self, train_loader, test_loader):
        """Apply the desired precision, split into batches and perform any other
        similar operations on the dataset.
        """
        train_set = list(train_loader)
        test_set = list(test_loader)

        # batch 0: all images, mode 0: data
        x = train_set[0][0].to(dtype=self.precision)
        x_t = test_set[0][0].to(dtype=self.precision)

        # batch 0: all images, mode 1: labels
        y = train_set[0][1]
        self.test_labels = test_set[0][1]

        self.train_data = torch.split(x, self.batch_size)
        self.train_labels = torch.split(y, self.batch_size)
        self.test_data = torch.split(x_t, 1)
        #self.test_labels = torch.split(y_t, self.batch_size)
        self.nb_batches = len(self.train_data)
        print ("Number of Batches: %d" %self.nb_batches)


    def step(self):
        """Loads a batch of images and labels.
        This method can be further customized to randomize the batch
        contents.
        """
        self.observation = self.train_data[self.current_batch_idx].cuda()
        self.labels = self.train_labels[self.current_batch_idx].cuda()
        self.current_batch_idx += 1
        if self.check_reset():
            self.reset()

#
