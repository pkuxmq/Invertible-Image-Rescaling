"""Model trainning & validating."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 30日 星期一 22:50:05 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import pdb

import torch
import torch.optim as optim

from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler

from data import get_data
from model import (get_model, model_device, model_load, model_save,
                   train_epoch, valid_epoch)

class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str,
                        default="output", help="output directory")
    parser.add_argument('--checkpoint', type=str,
                        default="models/ImageZoom_X4.pth", help="checkpoint file")
    parser.add_argument('--scale', type=int, default=4, help="scale factor")
    parser.add_argument('--bs', type=int, default=3, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    if (args.scale == 2):
        args.checkpoint = "models/ImageZoom_X2.pth"

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # get model
    model = get_model(args.scale)
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    print(model)

    # construct optimizer and learning rate scheduler,
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.99))

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = MultiStepLR_Restart(optimizer, [100000, 200000, 300000, 400000],
        restarts=[0],weights=[1],gamma=0.5,clear_state=False)

    # get data loader
    train_dl, valid_dl = get_data(trainning=True, bs=args.bs)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {} ...".format(epoch +
                                                          1, args.epochs, lr_scheduler.get_last_lr()))
        train_epoch(train_dl, model, optimizer, device, args.scale, tag='train')

        valid_epoch(valid_dl, model, device, args.scale, tag='valid')

        lr_scheduler.step()

        if (epoch + 1) % 100 == 0 or (epoch == args.epochs - 1):
            model_save(model, os.path.join(
                args.outputdir, "ImageZoom_X{}_{}.pth".format(args.scale, epoch + 1)))
