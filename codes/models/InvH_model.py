import logging
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss

logger = logging.getLogger('base')

class InvHSRModel(BaseModel):
    def __init__(self, opt):
        super(InvHSRModel, self).__init__(opt)

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

        self.upscale_log = int(math.log(opt['scale'], 2))

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

    #def loss_backward(self, x, y):
    #    x_samples = self.netG(y, rev=True)
    #    x_samples_image = x_samples[:, :3, :, :]
    #    l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

    #    return l_back_rec


    def optimize_parameters(self, step):
        if self.train_opt['stage1']:
            stage1_step = self.train_opt['stage1']
        else:
            stage1_step = 0

        op_num = self.opt['network_G']['block_num']
        fix_n = np.sum(op_num[:-1]) + len(op_num[:-1])
        if step <= stage1_step:
            for n, p in self.netG.named_parameters():
                index = int(n.split('.')[2])
                if 'haar' in n or index >= fix_n:
                    continue
                else:
                    p.requires_grad = False
        elif step == stage1_step + 1:
            for n, p in self.netG.named_parameters():
                index = int(n.split('.')[2])
                if 'haar' in n or index >= fix_n:
                    continue
                else:
                    p.requires_grad = True

        self.optimizer_G.zero_grad()

        self.input = [self.real_H]

        self.output = self.netG(self.input)
        loss = 0
            
        zshapes = []
        # forward loss
        l_forw_fit = 0
        l_forw_mle = 0
        for i in range(self.upscale_log):
            xx = self.output[i]
            z = xx[:, 3:, :, :]
            zshapes.append(z.shape)
            # mle
            l_forw_mle += self.train_opt['lambda_mle_forw'] * torch.sum(torch.norm(z, p=2, dim=1))
        # fit
        l_forw_fit += self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(self.output[-1][:, :3, :, :], self.var_L)

        if not self.train_opt['multi_reconstruction']:
            input_rev = []
            for zshape in zshapes:
                input_rev.append(self.noise_batch(zshape))
            input_rev.append(self.output[-1][:, :3, :, :])

            self.x_sr = self.netG(input_rev, rev=True)

            # backward loss
            l_back_rec = 0
            if step <= stage1_step:
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(self.output[-2][:, :3, :, :], self.x_sr[0])
            else:
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(self.x_sr[-1], self.real_H)
        else:
            input_rev = []
            for zshape in zshapes:
                input_rev.append(self.noise_batch(zshape))
            LRs = []
            for i in range(len(self.output)):
                LRs.append(self.output[i][:, :3, :, :])
            input_rev.append(LRs)

            self.x_sr = self.netG(input_rev, rev=True, multi_reconstruction=True)

            # backward loss
            l_back_rec = 0
            if step <= stage1_step:
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(self.output[-2][:, :3, :, :], self.x_sr[0])
            else:
                HRs = self.x_sr[-1]
                for i in range(len(HRs)):
                    l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(HRs[i], self.real_H)

        loss = l_forw_fit + l_forw_mle + l_back_rec

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
        self.input = [self.real_H]

        zshapes = []

        # forward
        self.netG.eval()
        with torch.no_grad():
            self.forw = self.netG(self.input)
        for i in range(self.upscale_log):
            zshapes.append(self.forw[i][:, 3:, :, :].shape)

        self.forw_L = self.forw[-1][:, :3, :, :]

        # backward
        input_rev = []
        for zshape in zshapes:
            input_rev.append(self.noise_batch(zshape))
        input_rev.append(self.forw[-1][:, :3, :, :])

        with torch.no_grad():
            self.x_sr = self.netG(input_rev, rev=True)

        self.fake_H = self.x_sr[-1]

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
