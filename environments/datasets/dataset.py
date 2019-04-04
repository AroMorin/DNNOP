"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
from ..environment import Environment

class Dataset(Environment):
    def __init__(self, env_params):
        print("Creating Dataset")
        super(Dataset, self).__init__(env_params)
        env_params = self.ingest_params_lvl1(env_params)
        self.data_path = env_params["data path"]
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
        self.set_optimization_mode()

    def ingest_params_lvl1(self, env_params):
        assert type(env_params) is dict
        default_params = {
                            "data path": "C:/Users/aaa2cn/Documents/mnist_data",
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

    def show_image(self):
        """Placeholder method to show a particular image of the training or
        test dataset.
        """
        pass
