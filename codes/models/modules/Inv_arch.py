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
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=5.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        #print(self)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s)

    def forward(self, x, rev=False):
        if math.isinf(torch.sum(x)):
            print('Get INF in the block input')
        if math.isnan(torch.sum(x)):
            print('Get NaN in the block input')

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            s1 = self.H(y1)
            self.s1 = s1
            t1 = self.G(y1)
            y2 = x2.mul(self.e(s1)) + t1
        else:
            s1 = self.H(x1)
            self.s1 = s1
            t1 = self.G(x1)
            y2 = (x2 - t1).div(self.e(s1))
            y1 = x1 - self.F(y2)

        y = torch.cat((y1, y2), 1)
        if math.isinf(torch.sum(y)):
            print('Get INF in the block output')
        if math.isnan(torch.sum(y)):
            print('Get NaN in the block output')

        #return torch.cat((y1, y2), 1)
        return y

    def jacobian(self, x, rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1), x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            jac = torch.sum(self.log_e(self.s1))
        else:
            jac = -torch.sum(self.log_e(self.s1))

        return jac


class InvBlockSigmoid(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlockSigmoid, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        #print(self)

    def forward(self, x, rev=False):
        if math.isinf(torch.sum(x)):
            print('Get INF in the block input')
        if math.isnan(torch.sum(x)):
            print('Get NaN in the block input')

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2.mul(torch.sigmoid(self.H(y1)) * 2) + self.G(y1)
        else:
            y2 = (x2 - self.G(x1)).div(torch.sigmoid(self.H(x1)) * 2 + 1e-6)
            y1 = x1 - self.F(y2)

        y = torch.cat((y1, y2), 1)
        if math.isinf(torch.sum(y)):
            print('Get INF in the block output')
        if math.isnan(torch.sum(y)):
            print('Get NaN in the block output')
            if rev:
                print('sigmoid: ' + str(torch.sigmoid(self.H(x1))))
            else:
                print('sigmoid: ' + str(torch.sigmoid(self.H(y1))))

        return torch.cat((y1, y2), 1)


class InvBlockExpSigmoid(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num):
        super(InvBlockExpSigmoid, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        #print(self)

    def forward(self, x, rev=False):
        if math.isinf(torch.sum(x)):
            print('Get INF in the block input')
        if math.isnan(torch.sum(x)):
            print('Get NaN in the block input')

        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            y2 = x2.mul(torch.exp(torch.sigmoid(self.H(y1)) * 2 - 1)) + self.G(y1)
        else:
            y2 = (x2 - self.G(x1)).div(torch.exp(torch.sigmoid(self.H(x1)) * 2 - 1))
            y1 = x1 - self.F(y2)

        y = torch.cat((y1, y2), 1)
        if math.isinf(torch.sum(y)):
            print('Get INF in the block output')
        if math.isnan(torch.sum(y)):
            print('Get NaN in the block output')
            if rev:
                print('sigmoid: ' + str(torch.sigmoid(self.H(x1))))
            else:
                print('sigmoid: ' + str(torch.sigmoid(self.H(y1))))

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
            i = 0
            for op in self.operations:
                out = op.forward(out, rev)
                i += 1
                print('forward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in forward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in forward block ' + str(i))
                    exit()
        else:
            i = 0
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                i += 1
                print('backward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in backward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in backward block ' + str(i))
                    exit()

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
            i = 0
            for op in self.operations:
                out = op.forward(out, rev)
                i += 1
                #print('forward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in forward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in forward block ' + str(i))
                    exit()
        else:
            i = 0
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                i += 1
                #print('backward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in backward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in backward block ' + str(i))
                    exit()

        return out


class InvSigmoidSRNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], upscale_log=2, shuffle_stage1=True):
        super(InvSigmoidSRNet, self).__init__()
        self.upscale_log = upscale_log

        operations = []

        channel_split_num1 = channel_in // 2
        for i in range(block_num[0]):
            b = InvBlockSigmoid(subnet_constructor, channel_in, channel_split_num1)
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
                b = InvBlockSigmoid(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ## initialization
        #mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        #if self.upscale == 4:
        #    mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            i = 0
            for op in self.operations:
                out = op.forward(out, rev)
                i += 1
                print('forward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in forward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in forward block ' + str(i))
                    exit()
        else:
            i = 0
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                i += 1
                print('backward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in backward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in backward block ' + str(i))
                    exit()

        return out


class InvExpSigmoidSRNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], upscale_log=2, shuffle_stage1=True):
        super(InvExpSigmoidSRNet, self).__init__()
        self.upscale_log = upscale_log

        operations = []

        channel_split_num1 = channel_in // 2
        for i in range(block_num[0]):
            b = InvBlockExpSigmoid(subnet_constructor, channel_in, channel_split_num1)
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
                b = InvBlockExpSigmoid(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

        ## initialization
        #mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        #if self.upscale == 4:
        #    mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x, rev=False):
        out = x

        if not rev:
            i = 0
            for op in self.operations:
                out = op.forward(out, rev)
                i += 1
                #print('forward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in forward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in forward block ' + str(i))
                    exit()
        else:
            i = 0
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                i += 1
                #print('backward sum ' + str(torch.sum(out)))
                if math.isinf(torch.sum(out)):
                    print('Get INF in backward block ' + str(i))
                    exit()
                if math.isnan(torch.sum(out)):
                    print('Get NaN in backward block ' + str(i))
                    exit()

        return out
