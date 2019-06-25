"""Class for applying perturbation."""

from __future__ import division
import numpy as np
import torch
import math
from torch.distributions import uniform, normal

class Noise(object):
    def __init__(self, hp, vector):
        self.hp = hp
        self.vec_length = vector.numel()
        print("Number of trainable parameters: %d" %self.vec_length)
        self.indices = np.arange(self.vec_length)
        self.running_idxs = np.arange(self.vec_length)
        self.noise_distribution = "uniform"  # Or "uniform"
        self.distribution = None
        self.choices = []  # list of indices
        self.limit = int(0.01*self.vec_length)
        self.lr = 0.2
        self.decay = 0.000012
        self.num_selections = 1000
        self.sr_min = -0.1
        self.sr_max = 0.1
        self.precision = vector.dtype
        self.vector = None
        self.counter = 1
        self.noise = None
        self.step = 0

    def update_state(self, integrity, p, improved):
        if self.step==0:
            #self.set_num_selections(integrity)
            self.set_choices(p)
            self.counter = 1
        if improved:
            self.step += int(100/self.counter)
            self.counter+=1
        self.set_sr(integrity)
        self.set_noise_dist()
        self.set_noise()
        self.set_vector()
        step = max(0, self.step-1)
        self.step = step
        print("Remaining steps: %d" %self.step)

    def set_num_selections(self, integrity):
        """Sets the number of selected neurons based on the integrity and
        hyperparameters."""
        p = 1.-integrity
        #p = integrity
        argument = (5*p)-3.5
        exp1 = math.tanh(argument)+1
        self.num_selections = max(1, int(exp1*0.5*self.limit))

    def set_sr(self, integrity):
        """Sets the search radius (noise magnitude) based on the integrity and
        hyperparameters."""
        p = integrity
        #p = 1.-integrity
        argument = (5*p)-2.0
        exp1 = math.tanh(argument)+1
        #self.sr_min = -exp1*0.05
        #self.sr_max = exp1*0.05
        lr = self.lr-self.decay
        self.lr = max(0.01, lr)
        self.sr_min = -exp1*self.lr
        self.sr_max = exp1*self.lr
        print("LR: %f " %self.lr)

    def set_noise_dist(self):
        """Determines the shape and magnitude of the noise."""
        a = self.sr_min
        b = self.sr_max
        c = (b-a)/2.
        assert a != b  # Sanity check
        if self.noise_distribution == "uniform":
            self.distribution = uniform.Uniform(torch.Tensor([a]), torch.Tensor([b]))
        elif self.noise_distribution == "normal":
            self.distribution = normal.Normal(torch.Tensor([c]), torch.Tensor([b]))
        else:
            print("Unknown distribution type")
            exit()

    def set_choices_(self, p):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        self.check_idxs()
        choices = np.random.choice(self.running_idxs, self.num_selections)
        self.running_idxs = np.delete(self.running_idxs, choices)
        self.choices = choices.tolist()

    def check_idxs(self):
        if len(self.running_idxs)<self.num_selections:
            self.running_idxs = np.arange(self.vec_length)

    def set_choices(self, p):
        """Use the numpy choices function (which has no equivalent in Pytorch)
        to generate a sample from the array of indices. The sample size and
        distribution are dynamically updated by the algorithm's state.
        """
        #p = p.cpu().numpy()  # Casting
        #self.choices = np.random.choice(self.indices, self.num_selections,
        #                                replace=False)
        choices = np.random.randint(0, self.vec_length, self.num_selections*2)
        choices = np.unique(choices)
        self.choices = choices[0:self.num_selections]

    def set_noise(self):
        noise = self.distribution.sample(torch.Size([self.num_selections]))
        # Cast to precision and CUDA, and edit shape
        self.noise = noise.to(dtype=self.precision, device='cuda').squeeze()

    def set_vector(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise_vector = torch.zeros(self.vec_length, dtype=self.precision,
                                    device='cuda')
        noise_vector[self.choices] = self.noise
        self.vector = noise_vector



#
