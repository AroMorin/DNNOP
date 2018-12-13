"""Base class for a Dataset. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
class Dataset:
    def __init__(self):
        self.batch_size = ''
        self.data_path = ''
        self.name = ''
        self.train_dataset = ''
        self.train_loader = ''
        self.train_tuples = ''
        self.train_set = ''
        self.train_data = ''
        self.train_labels = ''
        self.test_dataset = ''
        self.test_loader = ''
        self.test_tuples = ''
        self.test_set = ''
        self.test_data = ''
        self.test_labels = ''
        self.transforms = ''
        self.dataset = ''

    def init_dataset(self):
        """Placeholder method for initializing the dataset"""
        pass

    def implement_transformations(self):
        """Placeholder method for transformations on the dataset"""
        pass

    def get_train_data(self):
        """Placeholder method for retrieving training data of the dataset"""
        pass

    def get_train_labels(self):
        """Placeholder method for retrieving training labels of the dataset"""
        pass

    def get_test_data(self):
        """Placeholder method for retrieving test data of the dataset"""
        pass

    def get_test_labels(self):
        """Placeholder method for retrieving test labels of the dataset"""
        pass
