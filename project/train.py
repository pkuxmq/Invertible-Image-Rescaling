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

from data import get_data
from model import (get_model, model_device, model_load, model_save,
                   train_epoch, valid_epoch)

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str,
                        default="output", help="output directory")
    parser.add_argument('--checkpoint', type=str,
                        default="models/ImageZoom.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # get model
    model = get_model()
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    print(model)

    # construct optimizer and learning rate scheduler,
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.1)

    # get data loader
    train_dl, valid_dl = get_data(trainning=True, bs=args.bs)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {} ...".format(epoch +
                                                          1, args.epochs, lr_scheduler.get_last_lr()))

        train_epoch(train_dl, model, optimizer, device, tag='train')

        valid_epoch(valid_dl, model, device, tag='valid')

        lr_scheduler.step()

        if (epoch + 1) % 100 == 0 or (epoch == args.epochs - 1):
            model_save(model, os.path.join(
                args.outputdir, "ImageZoom_{}.pth".format(epoch + 1)))
