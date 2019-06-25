"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
from ..environment import Environment
import matplotlib.pyplot as plt
import torch


class Dataset(Environment):
    def __init__(self, env_params):
        print("Creating Dataset")
        super(Dataset, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.data_path = env_params["data path"]
        self.normalize = env_params["normalize"]
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.transforms = None
        self.train_data = [] # Entire set of training images
        self.test_data = [] # Entire set of test images
        self.train_labels = [] # Entire set of training labels
        self.test_labels = [] # # Entire set of test labels
        self.labels = [] # Current batch of labels
        self.batch_size = 0
        self.nb_batches = 0
        self.current_batch_idx = 0

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "data path": "C:/Users/aaa2cn/Documents/mnist_data",
                            "normalize": True
                            }
        default_params.update(env_params)  # Update with user selections
        return default_params

    def load_dataset(self):
        """Placeholder method for initializing and loading the dataset."""
        pass

    def set_transformations(self):
        """Placeholder method to define the desired transformations on the
        dataset.
        """
        pass

    def format_data(self):
        """Placeholder method for retrieving performing formatting and adjustments
        to the dataset.
        """
        pass

    def set_precision(self, precision=torch.float):
        """In case the user wanted to change the precision after loading the
        dataset."""
        self.precision = precision

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
