"""This script allows making objects of different dataset classes"""

from .rastrigin import Rastrigin
from .bukin6 import Bukin6
from .ackley import Ackley
from .easom import Easom
from .schwefel import Schwefel
from .rosenbrock import Rosenbrock
from .eggholder import Eggholder

def make_function(name, env_params):
    """Class factory method. This method takes the name of the desired
    function and returns an object of the corresponding class.
    """
    if name == 'rastrigin':
        return Rastrigin(env_params)
    elif name == 'ackley':
        return Ackley(env_params)
    elif name == 'bukin6':
        return Bukin6(env_params)
    elif name == 'easom':
        return Easom(env_params)
    elif name == 'eggholder':
        return Eggholder(env_params)
    elif name == 'schwefel':
        return Schwefel(env_params)
    elif name == 'rosenbrock':
        return Rosenbrock(env_params)
    else:
        print("Unknown function requested")
        exit()
    
