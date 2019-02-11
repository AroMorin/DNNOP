"""This script allows making objects of different dataset classes"""

from .rastrigin import Rastrigin

def make_function(name, plot, precision):
    """Class factory method. This method takes the name of the desired
    dataset and returns an object of the corresponding class.
    """
    if name == 'rastrigin':
        return Rastrigin(plot, precision)
    else:
        print("Unknown function requested")
        exit()
