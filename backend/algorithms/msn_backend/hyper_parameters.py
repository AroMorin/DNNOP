"""This class defines a dictionary that contains the default values for
hyper parameters. The object of this class thus encapsulates the entirety
of hyper parameters for the MSN algorithm, and facilitates their updates.
"""

class Hyper_Parameters:
    def __init__(self, hyper_params=None):
        """In here, I set minimization mode only and no maximization mode in order
        to reduce the chance of conflict. The user may remember to turn on
        min mode but forget to turn off max mode, and vice versa. Of course,
        there can be assert checks, but I'll just be inviting bugs for no reason.
        """
        print("Iniitializing hyper parameters of MSN")
        self.hyper_params = {
        "number of anchors": 4,
        "number of probes per anchor": 8,
        "pool size": 50,
        "learning rate": 1,
        "alpha":1,
        "beta": 1,
        "lambda":1,
        "minimum distance": 10,
        "minimum entropy": 0.01,
        "step size": 0.05,
        "minimum integrity": 0,
        "maximum integrity": 1,
        "minimization mode": True,
        "target": 0
        }
        # Update dictionary if appropriate
        if hyper_params != None:
            self.update_hyper_params_dict(hyper_params)
        # Instantiate hyper parameters for MSN algorithm
        self.nb_anchors = self.hyper_params["number of anchors"]
        self.nb_probes = self.hyper_params["number of probes per anchor"]
        self.lr = self.hyper_params["learning rate"]
        self.alpha = self.hyper_params["alpha"]
        self.beta = self.hyper_params["beta"]
        self.lambda = self.hyper_params["lambda"]
        self.min_dist = self.hyper_params["minimum distance"]
        self.min_entropy = self.hyper_params["minimum entropy"]
        self.step_size = self.hyper_params["step size"]
        self.def_integrity = self.hyper_params["default integrity"]
        self.min_integrity = self.hyper_params["minimum integrity"]
        self.max_integrity = self.hyper_params["maximum integrity"]
        self.target = self.hyper_params["target"]
        self.minimizing = self.hyper_params["minimization mode"]

    def update_hyper_params_dict(self, hyper_params):
        """This function updates the default hyper parameters dictionary. It
        expects a dictionary of hyper parameters. The loop traverses the given
        dictionary and updates the class's default dictionary with the new values.
        """
        for key in hyper_params:
            self.hyper_params[key] = hyper_params[key]
