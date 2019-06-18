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
        self.set_value()

    def set_value(self):
        self.value = 0.

    def update_state(self, observation, inference):
        self.x_0 = inference


#
