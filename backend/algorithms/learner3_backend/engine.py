"""base class for pool
Its functions are:
1) Initializing the pool with the given random conditions
2) Returning/Setting the pool
3) Sorting the pool members according to their performance
4) Maintain the pool composition of Elite, Anchors and Probes


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

    def step(self):
        self.prep_new_model(self.env.observation, self.env.labels,
                                    self.inference, self.score)
        self.generate()
        #self.mem()

    def prep_new_model(self, observation, label, inference, score):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        self.inference = inference
        #self.mem.update_state(observation, label, inference, score)
        self.analyzer.analyze(score)
        self.score = self.analyzer.score
        self.elite.set_elite(self.model, self.vector, self.inference, self.score)
        # Define noise magnitude and scale
        self.perturb.update_state(self.analyzer)

    def generate(self):
        self.probes.generate(self.elite.vector, self.perturb)
        self.vector = self.probes.vector
        self.update_model(self.vector)
        #time.sleep(0.5)

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
