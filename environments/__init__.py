"""This script allows making objects of different environment classes."""

import environments.datasets as dataset
import environments.functions as funcs

def make_env(type,
            name,
            plot = False,
            data_path='',
            batch_size=0,
            precision=None,
            loss=True):
    """Class factory method. This method takes the name of the environment
    and returns an object of the corresponding class (by going through the
    environment's own class factory).
    """
    if type == 'dataset':
        return dataset.make_dataset(name, data_path, batch_size, precision, loss)
    elif type == 'function':
        return funcs.make_function(name, plot, precision, data_path)

#
