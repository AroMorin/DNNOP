"""This script allows making objects of different dataset classes"""

from .atari import Atari
from .roboschool import Roboschool

def make_env(name, env_params):
    """Class factory method. This method takes the name of the desired
    nao robot function and returns an object of the corresponding class.
    """
    if name == 'atari':
        return Atari(env_params)
    elif name == 'roboschool':
        return Roboschool(env_params)
    else:
        print("Unknown openAI Gym env requested")
        exit()
