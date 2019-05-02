"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as MNIST_CNN
from .cnn_fashionMNIST import Net as FashionMNIST_CNN
from .mnist_cnn_msn import Net as MNIST_CNN_MSN
from .func_fc import Net as FUNC_FC
from .nao_fc import Net as NAO_FC
from .object_detection_msn import Net as OD_CNN_MSN
from .dqn import Net as DQN
from .dqn_ram import Net as DQN_RAM

import torch
import torch.nn as nn
import numpy as np

def make_model(name, model_params={}):
    """Makes a single model."""
    model_params = ingest_params(model_params)
    if model_params["grad"]:
        model = pick_model(name, model_params)
    else:
        with torch.no_grad():
            model = pick_model(name, model_params)
    model.cuda().to(model_params["precision"])
    init_weights(model, model_params["weight initialization scheme"])
    if model_params["pre-trained"]:
        load_weights(model, model_params["elite path"])
    return model

def make_pool(name, model_params={}):
    """Makes a pool of models, without the "grad" parameters since no gradient
    is calculated when a pool is used (ie. evolutionary algorithms don't need
    to calculate gradients).
    """
    model_params = ingest_params(model_params)
    pool = []
    for _ in range(model_params["pool size"]):
        with torch.no_grad():
            model = pick_model(name, model_params)
            model.cuda().to(model_params["precision"])
            init_weights(model, model_params["weight initialization scheme"])
            pool.append(model)
    assert len(pool) == model_params["pool size"]  # Sanity check
    return pool

def ingest_params(user_params):
    """Creates a default parameters dictionary, overrides it if necessary
    with user selections and then returns the result.
    """
    default_params = {
                    "pool size": 50,
                    "precision": torch.float,
                    "weight initialization scheme": "Default",
                    "pre-trained": False,
                    "elite path": 'C:/model_elite.pth',
                    "grad": False
                    }
    default_params.update(user_params)  # Override with user choices
    return default_params

def pick_model(name, model_params):
    """Defines which class of models to pick, based on user input."""
    if name == "MNIST CNN":
        model = MNIST_CNN(model_params)
    elif name == "MNIST CNN MSN":
        model = MNIST_CNN_MSN(model_params)
    elif name == "FashionMNIST CNN":
        model = FashionMNIST_CNN(model_params)
    elif name == "Function FC model":
        model = FUNC_FC(model_params)
    elif name == "NAO FC model":
        model = NAO_FC(model_params)
    elif name == "OD CNN MSN":
        model = OD_CNN_MSN(model_params)
    elif name == "DQN RAM model":
        model = DQN_RAM(model_params)
    elif name == "DQN model":
        model = DQN(model_params)
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

def load_weights(model, path):
    model.load_state_dict(torch.load(path))
