"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
from ..environment import Environment

class Dataset(Environment):
    def __init__(self, data_path, batch_size, precision, training_size=0, test_size=0):
        print("Creating Dataset")
        super().__init__(precision)
        self.training_size = training_size # Size of training set
        self.test_size = test_size # Size of validation set
        if batch_size == None:
            self.batch_size = self.training_size
        else:
            assert isinstance(batch_size, int) # Sanity check
            self.batch_size = batch_size
        assert isinstance(data_path, str) # Sanity check
        self.data_path = data_path
        self.train_dataset = ''
        self.test_dataset = ''
        self.train_loader = ''
        self.test_loader = ''
        self.train_data = '' # Entire set of training images
        self.train_labels = '' # Entire set of training labels
        self.x = '' # Current batch of training images
        self.y = '' # Current batch of training labels
        self.x_t = '' # Test images
        self.y_t = '' # Test labels
        self.transforms = ''
        self.nb_batches = 0
        self.current_batch_idx = 0

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
