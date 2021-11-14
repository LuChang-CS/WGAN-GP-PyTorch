from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def infinite_dataloader(dataloader):
    while True:
        for x, _ in dataloader:
            yield x


def load_mnist(batch_size, dataset_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.MNIST(dataset_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_cifar10(batch_size, dataset_path):
    pass


def get_train_test_loader(dataset, batch_size, dataset_path):
    if dataset == 'mnist':
        return load_mnist(batch_size, dataset_path)
    elif dataset == 'cifar10':
        return load_cifar10(batch_size, dataset_path)
    else:
        raise KeyError('{key} is not in supported datasets'.format(key=dataset))
