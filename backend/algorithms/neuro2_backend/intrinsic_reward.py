"""Base class for intrinsic reward."""
import torch

class IR(object):
    def __init__(self, hp):
        self.hp = hp
        self.x_0 = None  # Previous observation
        self.x_1 = None  # Previous action
        self.mismatch = 0.
        self.value = 0.

    def compute(self, observation, inference):
        self.x_1 = inference
        if self.x_0 is not None:
            self.mismatch = self.canberra_distance()
            self.set_value()
        self.update_state(observation, inference)

    def canberra_distance(self):
        a = self.x_0
        b = self.x_1
        x = a.sub(b).abs()
        y = torch.add(a.abs(), b.abs())
        f = torch.div(x, y)
        j = torch.masked_select(f, torch.isfinite(f))
        result = j.sum()
        return result.item()

    def set_value(self):
        self.value = min(self.mismatch*1, 1)

    def update_state(self, observation, inference):
        self.x_0 = inference


#
