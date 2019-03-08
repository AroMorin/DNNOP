"""This script allows making objects of different dataset classes"""

from .pose_assumption import Pose_Assumption

def make_nao(name, env_params):
    """Class factory method. This method takes the name of the desired
    nao robot function and returns an object of the corresponding class.
    """
    if name == 'pose assumption':
        return Pose_Assumption(env_params)
    else:
        print("Unknown nao robot function requested")
        exit()
