import os
import math

import imageio
import numpy as np
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


def im2uint8(images):
    if isinstance(images.flatten()[0], np.floating):
        images = (255.99 * images).astype('uint8')
    return images


class Logger:
    def __init__(self, path, generator, image_save_number):
        self.path = path

        self.d_loss = []
        self.g_loss = []
        self.w_distance = []

        self.train_ys = [self.d_loss, self.g_loss, self.w_distance]
        self.train_titles = ['Discriminator Loss', 'Generator Loss', 'Wasserstein Distance']

        self.test_d_loss = []
        self.test_title = 'Test Discriminator Loss'

        self.generator = generator
        self.noise = generator.get_noise(image_save_number)
        self.image_save_number = image_save_number

        self.gif_frame = []

    def add_train_point(self, d_loss, g_loss, w_distance):
        self.d_loss.append(d_loss)
        self.g_loss.append(g_loss)
        self.w_distance.append(w_distance)

    def add_test_point(self, test_d_loss):
        self.test_d_loss.append(test_d_loss)

    def plot_train(self):
        x = len(self.d_loss)
        for y, title in zip(self.train_ys, self.train_titles):
            plt.clf()
            plt.plot(np.arange(1, x + 1), y)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(title)
            plt.savefig(os.path.join(self.path, title.replace(' ', '_') + '.png'))

    def plot_test(self):
        step = len(self.d_loss) // len(self.test_d_loss)
        plt.clf()
        plt.plot(np.arange(1, len(self.test_d_loss) + 1) * step, self.test_d_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Test Discriminator Loss')
        plt.savefig(os.path.join(self.path, self.test_title.replace(' ', '_') + '.png'))

    def save_gif(self):
        samples = self.generator.sample(self.image_save_number, noise=self.noise, standardize=True)
        grid = make_grid(samples, nrow=int(math.sqrt(self.image_save_number)))
        grid = grid.cpu().numpy().transpose(1, 2, 0)
        self.gif_frame.append(im2uint8(grid))
        imageio.mimsave(os.path.join(self.path, 'transition.gif'), self.gif_frame)

    def save_images(self, iteration):
        samples = self.generator.sample(self.image_save_number, standardize=True)
        grid = make_grid(samples, nrow=int(math.sqrt(self.image_save_number)))
        grid = im2uint8(grid.cpu().numpy().transpose(1, 2, 0))
        imageio.imwrite(os.path.join(self.path, 'iteration_{i}.png'.format(i=iteration)), grid)
