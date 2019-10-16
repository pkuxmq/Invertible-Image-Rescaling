import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'inv-sr':
        from .Inv_model import InvSRModel as M
    elif model == 'invgan-sr':
        from .InvGAN_model import InvGANSRModel as M
    elif model == 'invganbi-sr':
        from .InvGANbi_model import InvGANbiSRModel as M
    elif model == 'invganforw-sr':
        from .InvGANforw_model import InvGANforwSRModel as M
    elif model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
