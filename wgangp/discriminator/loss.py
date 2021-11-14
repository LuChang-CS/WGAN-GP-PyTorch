import torch
from torch import nn, autograd


class WGANGPLoss(nn.Module):
    def __init__(self, discriminator, lambda_=10, k=1):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_ = lambda_
        self.k = k

    def forward(self, real_data, fake_data):
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data)
        gradient_penalty = self.get_gradient_penalty(real_data, fake_data)
        wasserstein_distance = d_real.mean() - d_fake.mean()
        d_loss = -wasserstein_distance + gradient_penalty
        return d_loss, wasserstein_distance

    def get_gradient_penalty(self, real_data, fake_data):
        batch_size = len(real_data)
        real_data = real_data.view(batch_size, -1)
        with torch.no_grad():
            alpha = torch.rand((batch_size, 1)).to(real_data.device)
            interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - self.k) ** 2).mean() * self.lambda_
        return gradient_penalty
