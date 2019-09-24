import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import models.modules.module_util as mutil


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlock, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        #print(self)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
        else:
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        #print(self)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2.mul(torch.exp(self.H(y1))) + self.G(y1)
        else:
            y2 = (x2 - self.G(x1)).div(torch.exp(self.H(x1)))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        #self.haar_weights = torch.ones(4, 1, 2, 2, dtype=torch.double)
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


class ShuffleChannel(nn.Module):
    def __init__(self, channel_num, channel_split_num):
        super(ShuffleChannel, self).__init__()
        self.channel_num = channel_num
        self.channel_split_num = channel_split_num

    def forward(self, x, rev=False):
        if not rev:
            return torch.cat((x[:, self.channel_split_num:, :, :], x[:, :self.channel_split_num, :, :]), 1)
        else:
            return torch.cat((x[:, self.channel_num - self.channel_split_num:, :, :], x[:, :self.channel_num - self.channel_split_num, :, :]), 1)


class InvSRNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], upscale_log=2, shuffle_stage1=False):
        super(InvSRNet, self).__init__()
        self.upscale_log = upscale_log

        operations = []

        channel_split_num1 = channel_in // 2
        for i in range(block_num[0]):
            b = InvBlock(subnet_constructor, channel_in, channel_split_num1)
            operations.append(b)
            if shuffle_stage1:
                if i != block_num[0] - 1 or block_num[0] % 2 == 0:
                    if i % 2 == 0:
                        operations.append(ShuffleChannel(channel_in, channel_split_num1))
                    else:
                        operations.append(ShuffleChannel(channel_in, channel_in - channel_split_num1))

        current_channel = channel_in
        for i in range(upscale_log):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i + 1]):
                b = InvBlock(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ## initialization
        #mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        #if self.upscale == 4:
        #    mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)

        return out


class InvExpSRNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], upscale_log=2, shuffle_stage1=True):
        super(InvExpSRNet, self).__init__()
        self.upscale_log = upscale_log

        operations = []

        channel_split_num1 = channel_in // 2
        for i in range(block_num[0]):
            b = InvBlockExp(subnet_constructor, channel_in, channel_split_num1)
            operations.append(b)
            if shuffle_stage1:
                if i != block_num[0] - 1 or block_num[0] % 2 == 0:
                    if i % 2 == 0:
                        operations.append(ShuffleChannel(channel_in, channel_split_num1))
                    else:
                        operations.append(ShuffleChannel(channel_in, channel_in - channel_split_num1))

        current_channel = channel_in
        for i in range(upscale_log):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(block_num[i + 1]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ## initialization
        #mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        #if self.upscale == 4:
        #    mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)

        return out
