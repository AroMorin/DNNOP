"""This script allows making objects of different solver classes"""

from .rl_solver import RL_Solver
from .func_solver import Func_Solver
from .dataset_solver import Dataset_Solver
from .robo_solver import Robo_Solver

def make_slv(name, slv_params):
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
    slv_params = ingest_params(slv_params)
    if name == 'dataset':
        return Dataset_Solver(slv_params)
    elif name == 'RL':
        return RL_Solver(slv_params)
    elif name == 'function':
        return Func_Solver(slv_params)
    elif name == 'robot':
        return Robo_Solver(slv_params)
    else:
        print("Unknown solver requested, exiting!")
        exit()

def ingest_params(slv_params):
    default_params = {
                        'environment': None,
                        'algorithm': None
                        }
    default_params.update(slv_params)
    assert default_params['environment'] is not None
    assert default_params['algorithm'] is not None
    return default_params


#
