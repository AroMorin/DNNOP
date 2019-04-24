"""This script allows making objects of different dataset classes"""

from .mspacman import MsPacman

def make_env(name, env_params):
    """Class factory method. This method takes the name of the desired
    nao robot function and returns an object of the corresponding class.
    """
    if name == 'MsPacman':
        return MsPacman(env_params)
    else:
        print("Unknown nao robot function requested")
        exit()
