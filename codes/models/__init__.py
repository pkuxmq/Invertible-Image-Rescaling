import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'IRN':
        from .Inv_model import InvSRModel as M
    elif model == 'IRN+':
        from .InvGAN_model import InvGANSRModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
