"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as CNN1
import torch

def make(name, precision=torch.float):
    if name == "MNIST CNN":
        model = CNN1()
        model = cast_model(model, precision)
        model.cuda()
        return model

def cast_model(model, precision):
    return model.to(precision)
