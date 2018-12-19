"""This script allows making objects of different algorithm classes"""

from .sgd import SGD
from .advanced_neuroevolution import Advanced_Neuroevolution

def make(name, model, optimizer=None):
    """Class factory method. This method takes the name of the desired
    algorithm and returns an object of the corresponding class.

    The variable "params" is a tuple of the relevant parameters. Position is
    significant and it's important to make sure that the correct parameter is
    in the correct position in the tuple object. This makes communication and
    extensibility easier.

    Since different algorithms can require vastly different sets of parameters,
    passing an abstract object in this level is extremely attractive.
    """
    if name == 'sgd':
        return SGD(model, optimizer)

    elif name == 'advanced_neuroevolution':
        return Advanced_Neuroevolution(model, optimizer)
