"""Base class for any environment in the framework. There are the common methods
presented as placeholders. Feel free to extend/implement them in the respective
class as needed.
In general, an environment has a step function to retrieve the next observation
for the solving algorithm to operate on.
"""
import torch

class Environment(object):
    def __init__(self, env_params):
        print("Initializing environment")
        self.precision = torch.float  # Default precision
        self.device = torch.device("cuda") # Always assume GPU training/testing
        self.score_type = "Loss"  # Assume environment will use loss as score
        self.loss = True  # Assume environments require loss
        self.loss_type = ''  # Environments that require loss define a loss type
        self.acc = False  # Use when the environment has an accuracy measure
        self.score = False  # Activate when the environment has an evaluation
        self.observation = None
        self.target = 0
        self.plot = False  # Maybe needed in some environments such as Functions
        self.minimize = True  # Is the target a minimum or a maximum?
        self.ingest_params_lvl0(env_params)

    def ingest_params_lvl0(self, env_params):
        if "precision" in env_params:
            self.precision = env_params["precision"]
        assert "score type" in env_params
        self.score_type = env_params["score type"]
        if "loss type" in env_params:
            self.set_scoring(env_params['loss type'])
        else:
            self.set_scoring()

    def set_scoring(self, loss_type=''):
        self.loss_type = loss_type
        if self.score_type == "Loss":
            self.loss = True
            self.acc = False
            self.score = False
        elif self.score_type == "Accuracy":
            self.loss = False
            self.acc = True
            self.score = False
        elif self.score_type == "Score":
            self.loss = False
            self.acc = False
            self.score = True
        elif self.score_type == "None":
            self.loss = False
            self.acc = False
            self.score = False
        else:
            print("Unknown scoring method")
            exit()

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
