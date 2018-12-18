"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.
"""
class Algorithm:
    def __init__(self, model, ):
        self.x = '' # Training data
        self.y = '' # Training labels/targets
        self.x_t = '' # Testing data
        self.y_t = '' # Testing labels/targets
        self.optimizer = '' # Optimizer for SGD-derivatives
        self.model = '' # A model to optimize
        self.nb_batches = '' # If data is separated into batches
        self.nb_test_batches = ''
        self.pool_size = '' # Size of pool, for pool-based algorithms


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
