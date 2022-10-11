import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet
from models.modules.RRDB import RRDBNet
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'
    if opt_net['gc']:
        gc = opt_net['gc']
    else:
        gc = 32

    use_ConvDownsampling = False
    down_first = False
    down_num = int(math.log(opt_net['scale'], 2))

    if which_model['use_ConvDownsampling']:
        use_ConvDownsampling = True
        down_first = True
        down_num = 1
    if which_model['down_first']:
        down_first = True

    netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init, gc=gc), opt_net['block_num'], down_num, use_ConvDownsampling=use_ConvDownsampling, down_first=down_first, down_scale=opt_net['scale'])

    return netG


def define_R(opt):
    opt_net = opt['network_R']
    return RRDBNet(opt_net['in_nc'], opt_net['out_nc'], opt_net['nf'], opt_net['nb'], opt_net['gc'])


# for Invertible decolorization-colorization
def define_grey(opt):
    opt_net = opt['network_grey']
    which_model = opt_net['which_model']
    rgb_type = which_model['rgb_type']
    subnet_type = which_model['subnet_type']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    Conv1x1Grey_learnable = True
    if which_model['Conv1x1Grey_learnable'] == False:
        Conv1x1Grey_learnable = False

    net_grey = InvGreyNet(rgb_type, subnet(subnet_type, init), opt_net['block_num'], Conv1x1Grey_learnable)

    return net_grey


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
