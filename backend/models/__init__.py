"""Class factory for different models, also ensures that the desired precision
is chosen.
"""

from .cnn_mnist import Net as MNIST_CNN
import torch

def make(name, precision=torch.float):
    if name == "MNIST CNN":
        model = MNIST_CNN()
        model.cuda().to(precision)
        return model
