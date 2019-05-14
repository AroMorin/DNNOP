"""Base class for any environment in the framework. There are the common methods
presented as placeholders. Feel free to extend/implement them in the respective
class as needed.
In general, an environment has a step function to retrieve the next observation
for the solving algorithm to operate on.
"""
import torch

class Environment(object):
    def __init__(self, env_params):
        print("Initializing environment!")
        env_params = self.ingest_params_lvl0(env_params)
        self.precision = env_params["precision"]
        self.device = torch.device("cuda") # Always assume GPU training/testing
        self.score_type = env_params["score type"]
        # Environments that require loss define a loss type
        self.loss_type = env_params["loss type"]
        self.loss = False
        self.acc = False
        self.score = False
        self.error = False
        self.observation = None
        self.labels = torch.zeros((1,), device='cuda', dtype=torch.float)
        self.target = 0
        self.minimize = True  # Are we trying to minimize a value, e.g. error?
        self.set_scoring(env_params)

    def ingest_params_lvl0(self, env_params):
        assert "score type" in env_params
        default_params = {
                            "precision": torch.float,
                            "loss type": '',
                            # only case when target is unknown is when the
                            # scoring mode is set to "score", we assume infinity
                            # unless the user defines something else
                            "target": float('inf')
                            }
        default_params.update(env_params)
        return default_params

    def set_scoring(self, env_params):
        if self.score_type == "loss":
            # Use when environment has a loss (differentiable) function
            self.loss = True
            self.acc = False
            self.score = False
            self.error = False
            self.minimize = True
            self.target = 0
        elif self.score_type == "error":
            # Activate when the environment has an evaluation function
            self.loss = False
            self.acc = False
            self.score = False
            self.error = True
            self.minimize = True
            self.target = 0
        elif self.score_type == "accuracy":
            # Use when the environment has an accuracy measure
            self.loss = False
            self.acc = True
            self.score = False
            self.error = False
            self.minimize = False  # Maximize accuracy
            self.target = 100.  # 100%
        elif self.score_type == "score":
            # Activate when the environment has an evaluation function
            self.loss = False
            self.acc = False
            self.score = True
            self.error = False
            self.minimize = False
            self.target = env_params["target"]  # User must define score
        elif self.score_type == "none":
            self.loss = False
            self.acc = False
            self.score = False
            self.error = False
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
