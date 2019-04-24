"""This script allows making objects of different environment classes."""

import environments.datasets as dataset
import environments.functions as funcs
import environments.nao as nao
import environments.tasks as tasks
import environments.openai as openai

def make_env(type, name, env_params):
    """Class factory method. This method takes the name of the environment
    and returns an object of the corresponding class (by going through the
    environment's own class factory).
    """
    if type == 'dataset':
        return dataset.make_dataset(name, env_params)
    elif type == 'function':
        return funcs.make_function(name, env_params)
    elif type == 'nao':
        return nao.make_nao(name, env_params)
    elif type == 'task':
        return tasks.make_task(name, env_params)
    elif type == 'openai':
        return tasks.make_env(name, env_params)
    else:
        print("Unknown environment type requested, exiting!")
        exit()
#
