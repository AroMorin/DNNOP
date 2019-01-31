"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as MNIST_CNN
from .mnist_cnn_msn import Net as MNIST_CNN_MSN
import torch
import torch.nn as nn

def make_model(name, precision=torch.float, init_scheme='Xavier Normal'):
    if name == "MNIST CNN":
        model = MNIST_CNN()
        model.cuda().to(precision)
        return model

def make_pool(name, pool_size, precision=torch.float, init_scheme='Xavier Normal'):
    pool = []
    for _ in range(pool_size):
        with torch.no_grad():
            if name == "MNIST CNN":
                model = MNIST_CNN()
            elif name == "MNIST CNN MSN":
                model = MNIST_CNN_MSN()
            else:
                print("Unknown model selected")
                exit()
            model.cuda().to(precision)
            model.apply(init_uniform)
            pool.append(model)
    assert len(pool) == pool_size
    return pool

def init_uniform(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight, a=-2, b=2)
