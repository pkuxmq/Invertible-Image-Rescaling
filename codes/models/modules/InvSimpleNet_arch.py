import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import models.modules.module_util as mutil
from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import *

def InvSimpleNet(channel_in=3, channel_out=3, block_num=[2, 12, 16], upscale_log=2):
    return InvSRNet(channel_in, channel_out, subnet('SimpleNet'), block_num, upscale_log=2)

