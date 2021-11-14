from torch import nn

from discriminator.discriminator import Discriminator


class Cifar10_Discriminator(Discriminator):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, images):
        pass
