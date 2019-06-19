"""base class for pool
The pool object will contain the models under optimization.
"""
from .noise import Noise
from .analysis import Analysis
from .integrity import Integrity
from .frustration import Frustration

class Engine(object):
    def __init__(self, hyper_params):
        self.analyzer = Analysis(hyper_params)
        self.integrity = Integrity(hyper_params)
        self.frustration = Frustration(hyper_params)

    def analyze(self, feedback, score):
        self.analyzer.analyze(feedback, score)
        #self.frustration.update(self.analyzer.analysis)

    def update_state(self):
        """Prepares the new pool based on the scores of the current generation
        and the results of the analysis (such as value of intergrity).
        """
        #self.integrity.set_integrity(self.analyzer.improved)
        # Define noise vector
        #self.noise.update_state(self.integrity.value)

    def update_table(self, model, feedback):
        observation, action, _ = feedback
        if self.analyzer.analysis == 'better':
            if not model.in_table:
                print("-----------------------------------")
                model.observations.append(observation)
                model.actions.append(action)

        if self.analyzer.analysis == 'worse':
            if model.in_table:
                del model.observations[model.idx]
                del model.actions[model.idx]

#
