"""This class defines a dictionary that contains the default values for
hyper parameters. The object of this class thus encapsulates the entirety
of hyper parameters for the MSN algorithm, and facilitates their updates.
"""

import math

class Hyper_Parameters:
    def __init__(self, hyper_params=None):
        """In here, I set minimization mode only and no maximization mode in order
        to reduce the chance of conflict. The user may remember to turn on
        min mode but forget to turn off max mode, and vice versa. Of course,
        there can be assert checks, but I'll just be inviting bugs for no reason.
        """
        print("Iniitializing hyper parameters of MSN")
        self.hyper_params = {}
        self.nb_anchors = 0
        self.nb_probes = 0
        self.pool_size = 0
        self.lr = 0
        self.alpha = 0
        self.beta = 0
        self.lambda_ = 0
        self.min_dist = 0
        self.min_entropy = 0
        self.step_size = 0
        self.patience = 0
        self.def_integrity = 0
        self.min_integrity = 0
        self.max_integrity = 0
        self.initial_integrity = 0
        self.target = 0
        self.minimizing = True
        self.initial_score = math.inf
        self.epsilon = 0.00000001  # Prevents division by zero
        self.set_hyperparams_dict()
        self.set_hyperparams(hyper_params)

    def set_hyperparams_dict(self, hyper_params):
        """This function updates the default hyper parameters dictionary. It
        expects a dictionary of hyper parameters. The loop traverses the given
        dictionary and updates the class's default dictionary with the new values.

        The assertion makes sure the user is not trying to edit a non-existent
        hyper parameter.
        """
        self.hyper_params = {
                                "number of anchors": 4,
                                "number of probes per anchor": 8,
                                "pool size": 50,
                                "learning rate": 1,
                                "alpha":1,
                                "beta": 1,
                                "lambda":1,
                                "patience": 80,
                                "minimum distance": 10,
                                "minimum entropy": 0.01,
                                "step size": 0.05,
                                "patience": 15,
                                "default integrity": 0.9,
                                "initial integrity": 0.5,
                                "minimum integrity": 0,
                                "maximum integrity": 0.9,
                                "minimization mode": True,
                                "target": 0,
                            }
        # Update dictionary if appropriate
        if isinstance(hyper_params, dict):
            # Loop over incoming hyper parameters dictionary
            for key in hyper_params:
                assert key in self.hyper_params
                self.hyper_params[key] = hyper_params[key]

    def set_hyperparams(self):
            # Instantiate hyper parameters for MSN algorithm
            self.nb_anchors = self.hyper_params["number of anchors"]
            self.nb_probes = self.hyper_params["number of probes per anchor"]
            self.pool_size = self.hyper_params["pool size"]
            self.lr = self.hyper_params["learning rate"]
            self.alpha = self.hyper_params["alpha"]
            self.beta = self.hyper_params["beta"]
            self.lambda_ = self.hyper_params["lambda"]
            self.min_dist = self.hyper_params["minimum distance"]
            self.min_entropy = self.hyper_params["minimum entropy"]
            self.step_size = self.hyper_params["step size"]
            self.patience = self.hyper_params["patience"]
            self.def_integrity = self.hyper_params["default integrity"]
            self.initial_integrity = self.hyper_params["initial integrity"]
            self.min_integrity = self.hyper_params["minimum integrity"]
            self.max_integrity = self.hyper_params["maximum integrity"]
            self.minimizing = self.hyper_params["minimization mode"]
            self.target = self.hyper_params["target"]
            self.set_initial_score()

    def set_initial_score(self):
        """By default we assume minimization, if not, then we switch the
        default score to negative infinity.
        """
        if not self.hyper_params['minimization mode']:
            self.initial_score = -math.inf
