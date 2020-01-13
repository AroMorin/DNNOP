"""This script allows making objects of different algorithm classes"""

from .sgd import SGD
from .msn import MSN
from .msn2 import MSN2
from .ls import LS
from .spiking1 import SPIKING1
from .spiking2 import SPIKING2
from .neuro1 import NEURO1
from .neuro2 import NEURO2
from .sar import SAR
from .random_search import RS

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

    elif name == 'random search':
        return RS(m, alg_params)

    elif name == 'MSN':
        return MSN(m, alg_params)
    elif name == 'MSN2':
        return MSN2(m, alg_params)

    elif name == 'local search':
        return LS(m, alg_params)

    elif name == 'spiking1':
        return SPIKING1(m, alg_params)
    elif name == 'spiking2':
        return SPIKING2(m, alg_params)

    elif name == 'neuro1':
        return NEURO1(m, alg_params)
    elif name == 'neuro2':
        return NEURO2(m, alg_params)

    elif name == 'sar':
        return SAR(m, alg_params)

    else:
        print("Unknown algorithm requested!")
        exit()
