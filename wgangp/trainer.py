import torch

from discriminator.loss import WGANGPLoss


OPTIMIZER = {
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop
}

supported_optimizers = OPTIMIZER.keys()


def get_optimizer(model, key, lr, **kwargs):
    if key not in supported_optimizers:
        raise KeyError('{key} optimizer is not supported.'.format(key=key))
    optimizer = OPTIMIZER[key](model.parameters(), lr=lr, **kwargs)
    return optimizer


class GeneratorTrainer:
    def __init__(self, generator, batch_size, train_num=1, optimizer='adam', lr=1e-3, **optim_kwargs):
        self.generator = generator
        self.batch_size = batch_size
        self.train_num = train_num
        self.optimizer = get_optimizer(self.generator, optimizer, lr, **optim_kwargs)

    def _step(self, discriminator):
        noise = self.generator.get_noise(self.batch_size)
        fake_data = self.generator(noise)

        output = discriminator(fake_data)
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def step(self, discriminator):
        self.generator.train()
        discriminator.eval()

        loss = 0
        for _ in range(self.train_num):
            loss += self._step(discriminator)
        loss /= self.train_num

        return loss


class DiscriminatorTrainer:
    def __init__(self, discriminator, batch_size, train_num=1, optimizer='adam', lr=1e-3, lambda_=10, k=1):
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.train_num = train_num
        self.optimizer = get_optimizer(self.discriminator, optimizer, lr)

        self.criterion = WGANGPLoss(discriminator, lambda_=lambda_, k=k)

    def _step(self, real_data, generator):
        fake_data = generator.sample(self.batch_size)
        loss, wasserstein_distance = self.criterion(real_data, fake_data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), wasserstein_distance.item()

    def step(self, real_data, generator):
        self.discriminator.train()
        generator.eval()

        loss, w_distance = 0, 0
        for _ in range(self.train_num):
            loss_i, w_distance_i = self._step(real_data, generator)
            loss += loss_i
            w_distance += w_distance_i
        loss /= self.train_num
        w_distance /= self.train_num

        return loss, w_distance

    def evaluate(self, data_loader, device):
        loss = 0
        for data, _ in data_loader:
            data = data.to(device)
            loss += self.discriminator(data).mean().item()
        loss = -loss / len(data_loader)
        return loss
