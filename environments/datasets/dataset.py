"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
from ..environment import Environment

class Dataset(Environment):
    def __init__(self, data_path, batch_size, precision, train_size, test_size):
        print("Creating Dataset")
        super().__init__(precision)
        self.train_size = train_size # Size of the training set
        self.test_size = test_size
        self.train_dataset = ''
        self.test_dataset = ''
        self.train_loader = ''
        self.test_loader = ''
        self.x = '' # Entire set of training images
        self.y = '' # Entire set of training labels
        self.train_data = '' # Current batch of training images
        self.train_targets = '' # Current batch of training labels
        self.test_data = '' # Test images (whole, not batches)
        self.test_targets = '' # Test labels (whole, not batches)
        self.transforms = ''
        self.nb_batches = 0
        self.batch_size = 0
        self.current_batch_idx = 0
        self.loss = True  # Whether this environment has a loss or not
        self.loss_type = 'NLL loss'
        self.set_batch_size(batch_size)
        assert isinstance(data_path, str)  # Sanity check
        self.data_path = data_path

    def set_batch_size(self, batch_size):
        if batch_size != 0:
            assert isinstance(batch_size, int)  # Sanity check
            self.batch_size = batch_size
        else:
            self.batch_size = self.train_size

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

    def show_image(self):
        """Placeholder method to show a particular image of the training or
        test dataset.
        """
        pass
