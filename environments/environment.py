"""Base class for any environment in the framework. There are the common methods
presented as placeholders. Feel free to extend/implement them in the respective
class as needed.
In general, an environment has a step function to retrieve the next observation
for the solving algorithm to operate on.
"""
import torch

class Environment(object):
    def __init__(self, precision=None):
        print("Initializing environment")
        self.precision = torch.float  # Default precision
        self.device = torch.device("cuda") # Always assume GPU training/testing
        self.loss = False  # assume environments require loss
        self.loss_type = ''  # environments that require loss define a loss type
        self.acc = False  # Use when the environment has an accuracy measure
        self.score = False  # Activate when the environment has an evaluation
        self.observation = None
        self.target = 0
        self.minimize = True  # Is the target a minimum or a maximum?
        self.set_precision(precision)

    def set_precision(self, precision):
        """Sets the precision of the environment."""
        if precision == None:
            return  # Do nothing
        else:
            self.precision = precision

    def step(self, alg=None):
        """Placeholder for step function. The step function is an essential
        component of any environment. It defines a "loading" of a batch of an
        arbitrary size (including 1) of data in context of a dataset.
        In essence, this method loads the next "observation" of the environment.
        It will have different meanings for different environments.
        """
        pass

    def reset(self):
        """Placeholder for method to reset the state of the environment."""
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
