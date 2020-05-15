import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
from models.modules.Apply_jpg import apply_jpg
import cv2

logger = logging.getLogger('base')

class InvSRModel(BaseModel):
    def __init__(self, opt):
        super(InvSRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()
        self.apply_jpg = apply_jpg()

        if self.is_train:
            self.netG.train()

            # loss
            if self.train_opt['pixel_criterion']:
                self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])
                self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])
            else:
                self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
                self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def noise_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    #def loss_forward(self, out, y):
    #    l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, :3, :, :], y)

    #    z = out[:, 3:, :, :].reshape([out.shape[0], -1])
    #    l_forw_mle = self.train_opt['lambda_mle_forw'] * torch.sum(torch.norm(z, p=2, dim=1))

    #    return l_forw_fit, l_forw_mle

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        #l_forw_mle = self.train_opt['lambda_mle_forw'] * torch.sum(torch.norm(z, p=2, dim=1))
        l_forw_mle = self.train_opt['lambda_mle_forw'] * torch.sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_mle

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        self.input = self.real_H

        self.output = self.netG(x=self.input)
        loss = 0
            
        zshape = self.output[:, 3:, :, :].shape

        #LR = self.output[:, :3, :, :]
        #LR = (LR * 255.).round() / 255.
        #LR = LR.detach()

        if self.train_opt['use_bicubic']:
            #LR = self.var_L
            LR_g = self.output[:, :3, :, :]
        elif (not self.train_opt['ignore_quantization']):
            #LR = self.Quantization(self.output[:, :3, :, :])
            LR_g = torch.clamp(self.output[:, :3, :, :], 0, 1)
            LR_g = self.Quantization(LR_g)

        else:
            #LR = self.output[:, :3, :, :]
            LR_g = torch.clamp(self.output[:, :3, :, :], 0, 1)

        #l_forw_fit, l_forw_mle = self.loss_forward(self.output, self.var_L)
        #if self.train_opt['apply_jpg']:
        #    quality = self.train_opt['jpg_quality'] if self.train_opt['jpg_quality'] else 95
        #    LRGT = self.apply_jpg(self.var_L, quality).detach()
        #else:
        #    LRGT = self.var_L.detach()
        LRGT = self.var_L.detach()

        l_forw_fit, l_forw_mle = self.loss_forward(LR_g, LRGT, self.output[:, 3:, :, :])

        if self.train_opt['use_bicubic']:
            LR = self.var_L
        else:
            LR = LR_g

        if self.train_opt['apply_jpg']:
            quality = self.train_opt['jpg_quality'] if self.train_opt['jpg_quality'] else 95
            LR = self.apply_jpg(LR, quality)
        yy = torch.cat((LR, self.noise_batch(zshape)), dim=1)

        l_back_rec = self.loss_backward(self.real_H, yy)

        loss += l_forw_fit + l_back_rec + l_forw_mle

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_forw_mle'] = l_forw_mle.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()

    def test(self):
        Lshape = self.var_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        noise_scale = 1

        if self.test_opt and self.test_opt['noise_scale'] != None:
            noise_scale = self.test_opt['noise_scale']

        self.netG.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            self.forw_L = torch.clamp(self.forw_L, 0, 1)

            self.forw_L = self.Quantization(self.forw_L)

            if self.test_opt and self.test_opt['apply_jpg']:
                quality = self.test_opt['jpg_quality'] if self.test_opt['jpg_quality'] else 95
                self.forw_L = self.apply_jpg(self.forw_L, quality)

        #self.forw_L = (self.forw_L * 255.).round() / 255.

        if self.test_opt and self.test_opt['use_bicubic']:
            y_forw = torch.cat((self.var_L, noise_scale * self.noise_batch(zshape)), dim=1)
        else:
            y_forw = torch.cat((self.forw_L, noise_scale * self.noise_batch(zshape)), dim=1)

        with torch.no_grad():
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
