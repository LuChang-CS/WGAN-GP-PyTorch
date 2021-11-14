from discriminator.mnist_discriminator import MNISTDiscriminator
from discriminator.cifar10_discriminator import Cifar10_Discriminator


def get_discriminator(dataset, device):
    if dataset == 'mnist':
        return MNISTDiscriminator().to(device)
    elif dataset == 'cifar10':
        return Cifar10_Discriminator().to(device)
    else:
        raise KeyError('{key} is not in supported discriminators'.format(key=dataset))
