"""This script allows making objects of different task environment classes"""

from .object_detection import Object_Detection

def make_task(name, env_params):
    """Class factory method. This method takes the name of the desired
    task and returns an object of the corresponding class.
    """
    if name == 'object detection':
        return Object_Detection(env_params)
    else:
        print("Unknown task environment requested")
        exit()
