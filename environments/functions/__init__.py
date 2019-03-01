"""This script allows making objects of different dataset classes"""

from .rastrigin import Rastrigin
from .bukin6 import Bukin6
from .ackley import Ackley
from .easom import Easom
from .schwefel import Schwefel
from .rosenbrock import Rosenbrock
from .eggholder import Eggholder

def make_function(name, plot, precision, data_path):
    """Class factory method. This method takes the name of the desired
    function and returns an object of the corresponding class.
    """
    if name == 'rastrigin':
        return Rastrigin(plot, precision, data_path)
    elif name == 'ackley':
        return Ackley(plot, precision, data_path)
    elif name == 'bukin6':
        return Bukin6(plot, precision, data_path)
    elif name == 'easom':
        return Easom(plot, precision, data_path)
    elif name == 'eggholder':
        return Eggholder(plot, precision, data_path)
    elif name == 'schwefel':
        return Schwefel(plot, precision, data_path)
    elif name == 'rosenbrock':
        return Rosenbrock(plot, precision, data_path)
    else:
        print("Unknown function requested")
        exit()
