from torch import nn

from generator.generator import Generator


class Cifar10Generator(Generator):
    def __init__(self, noise_dim, device):
        super().__init__(noise_dim, device)

        self.latent_dim = 128

        self.linear = nn.Sequential(
            nn.Linear(noise_dim, 4 * 4 * 4 * self.latent_dim),
            nn.BatchNorm1d(4 * 4 * 4 * self.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * self.latent_dim, out_channels=2 * self.latent_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(2 * self.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * self.latent_dim, out_channels=self.latent_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=3, kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

        self.output_dim = 3072

    def forward(self, noise):
        output = self.linear(noise)
        output = output.view(-1, 4 * self.latent_dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.tanh(output)
        output = output.view(-1, self.output_dim)
        return output

    def standardize(self, samples):
        samples = samples.view(-1, 3, 32, 32)
        samples = samples.mul(0.5).add(0.5)
        return samples
