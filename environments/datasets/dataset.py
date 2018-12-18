"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
class Dataset:
    def __init__(self, batch_size, data_path, precision):
        self.batch_size = batch_size
        self.data_path = data_path
        self.precision = torch.float
        self.device = torch.device("cuda") # Always assume GPU training/testing
        self.train_dataset = ''
        self.test_dataset = ''
        self.train_loader = ''
        self.test_loader = ''
        self.train_data = ''
        self.test_data = ''
        self.train_labels = ''
        self.test_labels = ''
        self.transforms = ''

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

    def set_precision(self):
        """Placeholder method to change the precision of the data set."""
        pass

    def show_image(self):
        """Placeholder method to show a particular image of the training or
        test dataset.
        """
        pass
