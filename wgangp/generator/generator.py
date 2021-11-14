import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim, device):
        super().__init__()
        self.noise_dim = noise_dim
        self.device = device

    def sample(self, batch_size, noise=None, standardize=False):
        with torch.no_grad():
            if noise is None:
                noise = self.get_noise(batch_size)
            samples = self.forward(noise)
            if standardize:
                samples = self.standardize(samples)
            return samples

    def standardize(self, samples):
        raise NotImplementedError

    def get_noise(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        return noise
