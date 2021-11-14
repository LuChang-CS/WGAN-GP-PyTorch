from torch import nn

from generator.generator import Generator


class Cifar10Generator(Generator):
    def __init__(self, noise_dim, device):
        super().__init__(noise_dim, device)
        pass

    def forward(self, noise):
        pass
