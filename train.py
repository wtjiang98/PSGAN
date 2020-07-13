import os

import argparse

from torch.backends import cudnn

from dataloder import get_loader
from psgan.solver import Solver
from setup import setup_config, setup_argparser


def train_net(config):
    # enable cudnn
    cudnn.benchmark = True

    data_loader = get_loader(config)
    solver = Solver(config, data_loader=data_loader, device="cuda")
    solver.train()

if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)

    train_net(config)
