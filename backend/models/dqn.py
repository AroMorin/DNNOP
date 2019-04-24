"""A script that defines a simple FC model for function solving"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()
        model_params = self.ingest_params_lvl1(model_params)
        self.conv1 = nn.Conv2d(model_params['in channels'], 32, kernel_size=8,
                                stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.act1 = nn.Relu()
        self.act2 = nn.Relu()
        self.act3 = nn.Relu()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        w1 = conv2d_size_out(model_params['w'], 8, 4)
        w2 = conv2d_size_out(w1, 4, 2)
        convw = conv2d_size_out(w2, 3, 1)

        h1 = conv2d_size_out(model_params['h'], 8, 4)
        h2 = conv2d_size_out(h1, 4, 2)
        convh = conv2d_size_out(h2, 3, 1)

        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.act4 = nn.Relu()
        self.fc2 = nn.Linear(512, model_params['number of outputs'])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x

    def ingest_params_lvl1(self, model_params):
        assert type(model_params) is dict
        default_params = {
                            "in channels": 3,
                            "number of outputs": 18,
                            "w": 210,
                            "h": 160
                            }
        default_params.update(model_params)  # Update with user selections
        return default_params
