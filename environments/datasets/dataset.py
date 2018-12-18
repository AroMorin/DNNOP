"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
class Dataset:
    def __init__(self):
        self.batch_size = ''
        self.data_path = ''
        self.train_dataset = ''
        self.test_dataset = ''
        self.train_loader = ''
        self.test_loader = ''
        self.train_data = ''
        self.test_data = ''
        self.train_labels = ''
        self.test_labels = ''
        self.transforms = ''
        self.precision = ''

    def load_dataset(self):
        """Placeholder method for initializing and loading the dataset."""
        pass

    def set_transformations(self):
        """Placeholder method to define the desired transformations on the
        dataset.
        """
        pass

    def get_train_set(self):
        """Placeholder method for retrieving training data & labels of the
        dataset.
        """
        pass

    def get_test_set(self):
        """Placeholder method for retrieving test data & labels of the
        dataset.
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
