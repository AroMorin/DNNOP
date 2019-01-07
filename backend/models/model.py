"""Base class for a pyTorch model. The reasoning for creating a separate class
file instead of defining the class as part of the environment is that
the same model can be re-used for different classes. Thus, this approach
eliminates repitition of model definition. Especially, by defining a base here
and then building models off that base should reduce code repition.

Honestly, in retrospect, I'm not sure what good this is. It seems that it is
better to have the entire model definition in one place so as to be a
self-contained solution, with instant import/export option to public domain
(regular PyTorch).
"""

from cnn_mnist import Net

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
