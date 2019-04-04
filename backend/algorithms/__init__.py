"""This script allows making objects of different algorithm classes"""

from .sgd import SGD
from .msn import MSN
from .learner import LEARNER

def make_alg(name, m, alg_params):
    """Class factory method. This method takes the name of the desired
    algorithm and returns an object of the corresponding class.

    The variable "hyper params" is a tuple of the relevant parameters. Position
    is significant and it's important to make sure that the correct parameter is
    in the correct position in the tuple object.

    "params" object refers to the neural network model(s) that will be optimized.
    In case of evolutionary algorithms, the params object is a list of models.
    In case of SGD/gradient-based algorithms, the param object is a single
    model.
    """
    if name == 'sgd':
        return SGD(m, alg_params)
    elif name == 'MSN':
        return MSN(m, alg_params)
    elif name == 'Learner':
        return LEARNER(m, alg_params)
    else:
        print("Unknown algorithm requested!")
        exit()
