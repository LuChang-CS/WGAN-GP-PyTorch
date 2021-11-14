import os

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for training WGAN-GP')
    parser.add_argument('--seed', default='6666', type=int)
    parser.add_argument('--dataset', default='mnist', type=str, choices=['mnist', 'cifar10'])
    parser.add_argument('--noise_dim', default=128, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--critic_iters', default=5, type=int)
    parser.add_argument('--generator_iters', default=1, type=int)
    parser.add_argument('--iters', default=200000, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--lambda_', default=10, type=float)
    parser.add_argument('--k', default=1, type=int)

    parser.add_argument('--dataset_path', default='dataset', type=str)
    parser.add_argument('--result_path', default='result', type=str)
    parser.add_argument('--test_freq', default='100', type=int)
    parser.add_argument('--save_freq', default='100', type=int)
    parser.add_argument('--image_save_number', default='64', type=int)

    args = parser.parse_args()
    return args


def get_paths(args):
    dataset_path = os.path.join(args.dataset_path, args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    result_path = os.path.join(args.result_path, args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fig_path = os.path.join(result_path, 'figs')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    params_path = os.path.join(result_path, 'params')
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    return dataset_path, fig_path, params_path
