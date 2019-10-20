import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlock, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
        else:
            y2 = x2 - self.G(x1)
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        return 0


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class InvBlockExp2(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp2, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        self.I = subnet_constructor(self.split_len2, self.split_len1)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            self.s1 = self.clamp * (torch.sigmoid(self.I(x2)) * 2 - 1)
            y1 = x1.mul(torch.exp(self.s1)) + self.F(x2)
            self.s2 = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s2)) + self.G(y1)
        else:
            self.s2 = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s2))
            self.s1 = self.clamp * (torch.sigmoid(self.I(y2)) * 2 - 1)
            y1 = (x1 - self.F(y2)).div(torch.exp(self.s1))

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s1) + torch.sum(self.s2)
        else:
            jac = -torch.sum(self.s1) - torch.sum(self.s2)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

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
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


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

    def jacobian(self, x, rev=False):
        return 0


class InvSRNet(nn.Module):
    def __init__(self, block_type, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], upscale_log=2, shuffle_stage1=False):
        super(InvSRNet, self).__init__()
        self.upscale_log = upscale_log

        operations = []

        channel_split_num1 = channel_in // 2
        for i in range(block_num[0]):
            if block_type == 'InvBlock':
                b = InvBlock(subnet_constructor, channel_in, channel_split_num1)
            elif block_type == 'InvBlockExp':
                b = InvBlockExp(subnet_constructor, channel_in, channel_split_num1)
            elif block_type == 'InvBlockExp2':
                b = InvBlockExp2(subnet_constructor, channel_in, channel_split_num1)
            else:
                print("Error! Undefined block type!")
                exit(1)
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
                if block_type == 'InvBlock':
                    b = InvBlock(subnet_constructor, current_channel, channel_out)
                elif block_type == 'InvBlockExp':
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                elif block_type == 'InvBlockExp2':
                    b = InvBlockExp2(subnet_constructor, current_channel, channel_out)
                else:
                    print("Error! Undefined block type!")
                    exit(1)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out
