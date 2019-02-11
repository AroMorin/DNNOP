"""Base class for any environment in the framework. There are the common methods
presented as placeholders. Feel free to extend/implement them in the respective
class as needed.
In general, an environment has a step function to retrieve the next observation
for the solving algorithm to operate on.
"""
import torch

class Environment:
    def __init__(self, precision=None):
        print("Initializing environment")
        self.device = torch.device("cuda") # Always assume GPU training/testing
        if precision == None:
            self.precision = torch.float
        else:
            self.precision = precision
        self.loss = False  # assume environments require loss
        self.loss_type = ''  # environments that require loss define a loss type
        self.acc = False  # Use when the environment has an accuracy measure
        self.score = False  # Activate when the environment has an evaluation
        self.observation = None
        self.target = 0
        self.minimize = True  # Is the target a minimum or a maximum?

    def step(self):
        """Placeholder for step function. The step function is an essential
        component of any environment. It defines a "loading" of a batch of data
        in context of a dataset. It will have different meanings for different
        environments.
        This method loads the next observation of the environment, essentially.
        """
        pass

    def reset(self):
        """Placeholder for method to reset the environment."""
        pass

    def check_reset(self):
        """Placeholder in case there needs to be a check before resetting the
        environment.
        """
        pass

    def set_precision(self, precision):
        """Placeholder method to change the precision of the data set."""
        pass




#
