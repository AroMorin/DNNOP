"""base class for pool
The pool object will contain the models under optimization.
"""
from .elite import Elite
from .probes import Probes
from .analysis import Analysis
from .memory import Memory
from .perturbation import Perturbation
from .converter import Converter

import time
import torch

class Engine(object):
    def __init__(self, model, hyper_params):
        self.analyzer = Analysis(hyper_params)
        self.elite = Elite(hyper_params)
        self.perturb = Perturbation(hyper_params)
        self.probes = Probes(hyper_params)
        self.mem = Memory(hyper_params)
        self.model = model
        self.weights = Weights(self.model.state_dict())
        self.perturb.init_perturbation(self.vector)

    def prep_new_model(self, observation, inference, score):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        #self.mem.update_state(observation, label, inference, score)
        self.analyzer.analyze(score)
        # Define noise magnitude and scale
        self.perturb.update_state(self.analyzer)

    def generate(self):
        self.probes.generate(self.elite.vector, self.perturb)
        self.weights.set_vector(self.probes.vector)
        self.weights.update_model()

    def evaluate(self):
        self.mem.evaluate_model(self.model)
        if not self.mem.desirable:
            self.analyzer.suspend_reality()
            self.perturb.suspend_reality()
        for i in range(100):
            if not self.mem.desirable:
                eval = torch.tensor(self.mem.eval, device='cuda', dtype=torch.float)
                self.analyzer.analyze(eval)
                #self.perturb.update_state(self.analyzer)
                self.generate()
                self.mem.evaluate_model(self.model)
            else:
                print("Expected: %f------------------------" %self.mem.eval)
                return
        self.analyzer.restore_reality()
        self.perturb.restore_reality()

    def prep_new_hypothesis(self, score):
        self.analyzer.analyze(score)
        # Define noise magnitude and scale
        self.perturb.update_state(self.analyzer)





#
