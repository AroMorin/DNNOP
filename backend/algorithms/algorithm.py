"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""
from .interrogator import Interrogator

class Algorithm(object):
    def __init__(self):
        self.interrogator = Interrogator()

    def optimize(self):
        """Placeholder method for performing an optimization step."""
        pass

    def reset_state(self):
        """Placeholder method in case the solver has need to reset its internal
        state.
        """
        pass

    def save_weights(self, models, path):
        for i, model in enumerate(models):
            fn = path+"model_"+str(i)+".pth"
            torch.save(model.state_dict(), fn)

    def save_elite_weights(self, model, path):
        fn = path+"model_elite.pth"
        torch.save(model.state_dict(), fn)

    def achieved_target(self):
        """Determines whether the algorithm achieved its target or not."""
        best = self.optim.pool.elite.elite_score
        if self.hyper_params.minimizing:
            return best <= (self.hyper_params.target + self.hyper_params.tolerance)
        else:
            return best >= (self.hyper_params.target - self.hyper_params.tolerance)






#
