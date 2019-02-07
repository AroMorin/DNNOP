"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
from ..environment import Environment

class Dataset(Environment):
    def __init__(self, data_path, precision, loss):
        print("Creating Dataset")
        super().__init__(precision)
        assert isinstance(data_path, str)  # Sanity check
        self.data_path = data_path
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
        self.set_optimization_mode(loss)

    def set_batch_size(self, batch_size):
        if batch_size != 0:
            assert isinstance(batch_size, int)  # Sanity check
            self.batch_size = batch_size
        else:
            self.batch_size = self.train_size

    def set_optimization_mode(self, loss):
        if loss:
            self.loss = True
            self.loss_type = 'NLL loss'
            self.acc = False
            self.minimize = True
            self.target = 0
        else:
            self.loss = False
            self.acc = True
            self.minimize = False
            self.target = 100.  # 100% accuracy

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
