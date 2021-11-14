from generator.mnist_generator import MNISTGenerator
from generator.cifar10_generator import Cifar10Generator


def get_generator(dataset, noise_dim, device):
    if dataset == 'mnist':
        return MNISTGenerator(noise_dim, device).to(device)
    elif dataset == 'cifar10':
        return Cifar10Generator(noise_dim, device).to(device)
    else:
        raise KeyError('{key} is not in supported generators'.format(key=dataset))
