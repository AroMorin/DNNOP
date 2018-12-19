"""Base class for a pyTorch model. The reasoning for creating a separate class
file instead of defining the class as part of the environment is that
the same model can be re-used for different classes. Thus, this approach
eliminates repitition of model definition. Especially, by defining a base here
and then building models off that base should reduce code repition.
"""

class Model(nn.Module):
    def __init__(self):
                super(Net, self).__init__()
