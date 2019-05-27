"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch

class Selection_P(object):
    def __init__(self, hp, length):
        self.hp = hp
        uniform_vec = torch.full((length,), 0.5, device='cuda')
        self.p = torch.nn.functional.softmax(uniform_vec, dim=0)
        self.f = 0.01

    def update_state(self, vec):
        p = self.get_p(vec)
        while self.zeros_present(p):
            f = self.f-0.001
            self.f = max(0.0001, f)
            p = self.get_p(vec)
        self.p = p.detach()
        self.reset()

    def get_p(self, vec):
        vec1 = vec.mul(self.f)
        return torch.nn.functional.softmax(vec1, dim=0)

    def zeros_present(self, v):
        num = v.eq(0).nonzero().size()[0]
        return num>0

    def reset(self):
        f = self.f+0.01
        self.f = min(10., f)
#
