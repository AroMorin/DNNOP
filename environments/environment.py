"""Base class for any environment in the framework. There are the common methods
presented as placeholders. Feel free to extend/implement them in the respective
class as needed.
In general, an environment has a step function to retrieve the next observation
for the solving algorithm to operate on.
"""
import torch

class Environment:
    def __init__(self, precision):
        print("Initializing environment")
        self.device = torch.device("cuda") # Always assume GPU training/testing
        self.precision = precision

    def step(self):
        """Placeholder for step function. The step function is an essential
        component of any environment. It defines a "loading" of a batch of data
        in context of a dataset. It will have different meanings for different
        environments.
        This method loads the next observation of the environment, essentially.
        """
        pass

    def check_reset(self):
        """Placeholder in case there needs to be a check before resetting the
        environment.
        """
        pass

    def reset(self):
        """Placeholder for method to reset a dataset. Useful after batches
        reach the end.
        """
        pass

    def set_precision(self):
        """Placeholder method to change the precision of the data set."""
        pass
