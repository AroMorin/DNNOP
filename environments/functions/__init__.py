"""This script allows making objects of different dataset classes"""

from .rastrigin import Rastrigin

def make_function(name, nb_dimensions, plot):
    """Class factory method. This method takes the name of the desired
    dataset and returns an object of the corresponding class.
    """
    if name == 'rastrigin':
        return Rastrigin(nb_dimensions, plot)
    else:
        print("Unknown function requested")
        exit()
