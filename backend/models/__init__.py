"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as MNIST_CNN
from .mnist_cnn_msn import Net as MNIST_CNN_MSN
from .basic_fc import Net as BASIC_FC
import torch
import torch.nn as nn
import numpy as np

def make_model(name, precision=torch.float, init_scheme='Default'):
    """Makes a single model."""
    model = pick_model(name)
    model.cuda().to(precision)
    init_weights(model, init_scheme)
    return model

def make_pool(name, pool_size, precision=torch.float, init_scheme='Identical'):
    """Makes a pool of models, without the "grad" parameters since no gradient
    is calculated when a pool is used (ie. evolutionary algorithms don't need
    to calculate gradients).
    """
    pool = []
    for _ in range(pool_size):
        with torch.no_grad():
            model = pick_model(name)
            model.cuda().to(precision)
            init_weights(model, init_scheme)
            pool.append(model)
    assert len(pool) == pool_size
    return pool

def pick_model(name):
    """Defines which class of models to pick, based on user input."""
    if name == "MNIST CNN":
        model = MNIST_CNN()
    elif name == "MNIST CNN MSN":
        model = MNIST_CNN_MSN()
    elif name == "Function FC model":
        model = BASIC_FC()
    else:
        print("Unknown model selected")
        exit()
    return model

def init_weights(model, scheme):
    """Initializes the weights of the model according to a defined scheme."""
    if scheme == 'Uniform':
        model.apply(init_uniform)
    elif scheme == 'Normal':
        model.apply(init_normal)
    elif scheme == 'Identical':
        model.apply(init_eye)
    else:
        return  # Default initialization scheme

def init_uniform(m):
    """Initializes weights according to a Uniform distribution."""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        limit = 0.1
        nn.init.uniform_(m.weight, a=-limit, b=limit)

def init_normal(m):
    """Initializes weights according to a Normal distribution."""
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        limit = 0.1
        origin = 0
        nn.init.normal_(m.weight, mean=origin, std=limit)

def init_eye(m):
    """Initializes weights according to an Identity matrix. This special case
    allows the initial input(s) to be reflected in the output of the model.
    """
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.eye_(m.weight)
