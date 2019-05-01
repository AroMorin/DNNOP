"""Base class for probes"""

from .perturbation import Perturbation
import torch

class Generator(object):
    def __init__(self, hp):
        self.vector = None

    def step(self, vector, perturb):
        """Set the new probes based on the calculated anchors."""
        probe = vector.clone()
        perturb.apply(probe)
        self.vector = probe

    def apply(self, vec):
        """Generate a list of random indices based on the number of selections,
        without duplicates.
        Then generate the noise vector with the appropriate size and range
        based on the search radius.
        Finally use the above to add to the vector of choice.
        """
        noise = self.get_noise()
        vec.add_(noise)

    def get_noise(self):
        """ This function defines a noise tensor, and returns it. The noise
        tensor needs to be the same shape as our originial vecotr. Hence, a
        "basis" tensor is created with zeros, then the chosen indices are
        modified.
        """
        noise = self.distribution.sample(torch.Size([self.size]))
        # Cast to precision and CUDA, and edit shape
        noise = noise.to(dtype=self.precision, device=self.device).squeeze()
        noise_vector = torch.zeros((self.vec_length), dtype=self.precision, device=self.device)
        noise_vector[self.choices] = noise
        return noise_vector


#
