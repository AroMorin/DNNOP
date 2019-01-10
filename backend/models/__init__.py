"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as MNIST_CNN
import torch

def make_model(name, precision=torch.float, init_scheme='Xavier Normal'):
    if name == "MNIST CNN":
        model = MNIST_CNN()
        model.cuda().to(precision)
        return model

def make_pool(name, pool_size, precision=torch.float, init_scheme='Xavier Normal'):
    pool = []
    for _ in range(pool_size):
        if name == "MNIST CNN":
            model = MNIST_CNN()
            model.cuda().to(precision)
            pool.append(model)
    assert len(pool) == pool_size
    return pool
