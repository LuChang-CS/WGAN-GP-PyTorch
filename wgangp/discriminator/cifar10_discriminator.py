from torch import nn

from discriminator.discriminator import Discriminator


class Cifar10_Discriminator(Discriminator):
    def __init__(self):
        super().__init__()

        self.latent_dim = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.latent_dim, out_channels=2 * self.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.latent_dim, out_channels=4 * self.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.fc = nn.Linear(4 * 4 * 4 * self.latent_dim, 1)

    def forward(self, images):
        images = images.view(-1, 3, 32, 32)

        output = self.block1(images)
        output = self.block2(output)
        output = self.block3(output)

        output = output.view(-1, 4 * 4 * 4 * self.latent_dim)
        output = self.fc(output)
        output = output.view(-1)
        return output
