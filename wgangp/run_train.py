import os
import random

import torch
import numpy as np

from config import parse_args, get_paths
from logger import Logger
from dataloader import get_train_test_loader, infinite_dataloader
from generator import get_generator
from discriminator import get_discriminator
from trainer import GeneratorTrainer, DiscriminatorTrainer


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_path, fig_path, params_path = get_paths(args)
    train_loader, test_loader = get_train_test_loader(args.dataset, args.batch_size, dataset_path)

    generator = get_generator(args.dataset, args.noise_dim, device)
    discriminator = get_discriminator(args.dataset, device)

    g_trainer = GeneratorTrainer(generator, args.batch_size, args.generator_iters, args.optimizer, args.lr)
    d_trainer = DiscriminatorTrainer(discriminator,
                                     args.batch_size, args.critic_iters,
                                     args.optimizer, args.lr,
                                     args.lambda_, args.k)

    train_gen = infinite_dataloader(train_loader)
    logger = Logger(fig_path, generator, args.image_save_number)
    for i in range(1, args.iters + 1):
        real_data = next(train_gen).to(device)

        d_loss, w_distance = d_trainer.step(real_data, generator)
        g_loss = g_trainer.step(discriminator)

        logger.add_train_point(d_loss, g_loss, w_distance)

        if i % args.test_freq == 0:
            test_d_loss = d_trainer.evaluate(test_loader, device)
            logger.add_test_point(test_d_loss)
            print('\r{} / {} iterations: D_Loss -- {:.6f} -- G_Loss -- {:.6f} -- W_dist -- {:.6f} -- Test_D_Loss -- {:6f}'
                  .format(i, args.iters, d_loss, g_loss, w_distance, test_d_loss))
        else:
            print('\r{} / {} iterations: D_Loss -- {:.6f} -- G_Loss -- {:.6f} -- W_dist -- {:.6f}'
                  .format(i, args.iters, d_loss, g_loss, w_distance), end='')

        if i % args.save_freq == 0:
            logger.plot_train()
            logger.plot_test()
            logger.save_gif()
            logger.save_images(i)

    torch.save(discriminator.state_dict(), os.path.join(params_path, 'discriminator.pt'))
    torch.save(generator.state_dict(), os.path.join(params_path, 'generator.pt'))


if __name__ == '__main__':
    args = parse_args()
    train(args)
