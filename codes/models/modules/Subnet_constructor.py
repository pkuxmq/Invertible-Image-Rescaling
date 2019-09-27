import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

class SimpleNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming'):
        super(SimpleNetBlock, self).__init__()

        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        body = []
        expand = 6

        body.append(nn.Conv2d(channel_in, channel_in * expand, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(channel_in * expand, channel_in, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(channel_in, channel_out, 3, padding=3//2))
        body.append(act)

        if init == 'xavier':
            mutil.initialize_weights_xavier(body[:-1], 0.1)
        else:
            mutil.initialize_weights(body[:-1], 0.1)
        mutil.initialize_weights(body[-1], 0)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class THNetBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming'):
        super(THNetBlock, self).__init__()

        act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        body = []
        hidden_channel = 64

        body.append(nn.Conv2d(channel_in, hidden_channel, 1, padding=1//2))
        body.append(act)
        body.append(nn.Conv2d(hidden_channel, hidden_channel, 3, padding=3//2))
        body.append(act)
        body.append(nn.Conv2d(hidden_channel, channel_out, 1, padding=1//2))
        body.append(act)

        if init == 'xavier':
            mutil.initialize_weights_xavier(body[:-1], 0.1)
        else:
            mutil.initialize_weights(body[:-1], 0.1)
        mutil.initialize_weights(body[-1], 0)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


def subnet(net_structure, init='kaiming'):
    def constructor(channel_in, channel_out):
        if net_structure == 'SimpleNet':
            if init == 'xavier':
                return SimpleNetBlock(channel_in, channel_out, init)
            else:
                return SimpleNetBlock(channel_in, channel_out)
        elif net_structure == 'THNet':
            if init == 'xavier':
                return THNetBlock(channel_in, channel_out, init)
            else:
                return THNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor
