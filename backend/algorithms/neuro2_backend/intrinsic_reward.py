"""Base class for elite."""
import torch

class IR(object):
    def __init__(self, hp):
        self.hp = hp
        self.ground_truth = None
        self.prediction = None
        self.mismatch = 0.
        self.value = 0.

    def compute(self, observation, inference):
        self.update_state(observation, inference)
        self.mismatch = self.canberra_distance()
        self.set_value()

    def update_state(self, observation, inference):
        self.ground_truth = observation
        self.prediction = inference
        self.mismatch = 0.
        self.value = 0.

    def canberra_distance(self):
        a = self.ground_truth
        b = self.prediction
        x = a.sub(b).abs()
        y = torch.add(a.abs(), b.abs())
        f = torch.div(x, y)
        j = torch.masked_select(f, torch.isfinite(f))
        result = j.sum()
        return result.item()

    def set_value(self):
        self.value = self.mismatch*0.5

#
