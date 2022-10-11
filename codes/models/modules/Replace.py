import torch
import torch.nn as nn

class replace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, replace_input):
        return replace_input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Replace(nn.Module):
    def __init__(self):
        super(Replace, self).__init__()

    def forward(self, input, replace_input):
        return replace.apply(input, replace_input)
