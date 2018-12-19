"""This script allows making objects of different environment classes"""

import environments.datasets as dataset
import environments.functions as funcs

def make(type, name, batch_size, data_path):
    """Class factory method. This method takes the name of the desired
    dataset and returns an object of the corresponding class.
    """
    if type == 'dataset':
        return dataset.make(name, batch_size, data_path)
    elif type == 'function':
        return dataset.make(name, batch_size, data_path)

#
