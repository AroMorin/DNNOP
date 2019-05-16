"""This script allows making objects of different dataset classes"""

from .atari import Atari

def make_env(name, env_params):
    """Class factory method. This method takes the name of the desired
    nao robot function and returns an object of the corresponding class.
    """
    if name == 'atari':
        return Atari(env_params)
    else:
        print("Unknown nao robot function requested")
        exit()
