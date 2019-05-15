"""This script allows making objects of different algorithm classes"""

from .sgd import SGD
from .msn import MSN
from .msn2 import MSN2
from .learner import LEARNER
from .learner2 import LEARNER2
from .learner3 import LEARNER3
from .learner4 import LEARNER4
from .learner5 import LEARNER5
from .learner6 import LEARNER6
from .learner7 import LEARNER7

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
    elif name == 'MSN2':
        return MSN2(m, alg_params)
    elif name == 'learner':
        return LEARNER(m, alg_params)
    elif name == 'learner2':
        return LEARNER2(m, alg_params)
    elif name == 'learner3':
        return LEARNER3(m, alg_params)
    elif name == 'learner4':
        return LEARNER4(m, alg_params)
    elif name == 'learner5':
        return LEARNER5(m, alg_params)
    elif name == 'learner6':
        return LEARNER6(m, alg_params)
    elif name == 'learner7':
        return LEARNER7(m, alg_params)
    else:
        print("Unknown algorithm requested!")
        exit()
