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


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='kaiming', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class CALayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(CALayer, self).__init__()
        # gloval acerage pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', kernel_size=3, reduction=3, bias=True, bn=False, act=nn.ReLU(True)):
        super(ChannelAttentionBlock, self).__init__()
        modules_body = []

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, 1, bias=bias)
        modules_body.append(self.conv1)
        if bn: modules_body.append(nn.BatchNorm2d(channel_in))
        modules_body.append(act)

        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, 1, bias=bias)
        modules_body.append(self.conv2)
        if bn: modules_body.append(nn.BatchNorm2d(channel_out))

        self.calayer = CALayer(channel_out, reduction)
        modules_body.append(self.calayer)

        self.body = nn.Sequential(*modules_body)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.calayer], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.calayer], 0.1)
        mutil.initialize_weights(self.conv2, 0)

    def forward(self, x):
        return self.body(x)

class CABlocks(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', kernel_size=3, reduction=3, bias=True, bn=False, act=nn.ReLU(True)):
        super(CABlocks, self).__init__()
        self.CABlock1 = ChannelAttentionBlock(channel_in, channel_in, init, kernel_size, reduction, bias, bn, act)
        self.CABlock2 = ChannelAttentionBlock(channel_in, channel_in, init, kernel_size, reduction, bias, bn, act)
        self.CABlock3 = ChannelAttentionBlock(channel_in, channel_out, init, kernel_size, reduction, bias, bn, act)

    def forward(self, x):
        out = self.CABlock1(x)
        out = self.CABlock2(out)
        out = self.CABlock3(out)

        return out


def subnet(net_structure, init='kaiming'):
    def constructor(channel_in, channel_out):
        if net_structure == 'SimpleNet':
            if init == 'xavier':
                return SimpleNetBlock(channel_in, channel_out, init)
            else:
                return SimpleNetBlock(channel_in, channel_out)
        elif net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'CABNet':
            if init == 'xavier':
                return CABlocks(channel_in, channel_out, init)
            else:
                return CABlocks(channel_in, channel_out)
        else:
            return None

    return constructor
