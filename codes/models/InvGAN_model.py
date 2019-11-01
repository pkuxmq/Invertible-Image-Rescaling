import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, ReconstructionLoss

logger = logging.getLogger('base')

class InvGANSRModel(BaseModel):
    def __init__(self, opt):
        super(InvGANSRModel, self).__init__(opt)

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

        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()

            # loss
            #if self.train_opt['pixel_criterion']:
            #    if self.train_opt['pixel_critetion'] == 'l2':
            #        self.Reconstruction_forw = nn.MSELoss().to(self.device)
            #        self.Reconstruction_back = nn.MSELoss().to(self.device)
            #    else:
            #        self.Reconstruction_forw = nn.L1Loss().to(self.device)
            #        self.Reconstruction_back = nn.L1Loss().to(self.device)
            #else:
            #    if self.train_opt['pixel_critetion_forw'] == 'l2':
            #        self.Reconstruction_forw = nn.MSELoss().to(self.device)
            #    else:
            #        self.Reconstruction_forw = nn.L1Loss().to(self.device)
            #    if self.train_opt['pixel_critetion_back'] == 'l2':
            #        self.Reconstruction_back = nn.MSELoss().to(self.device)
            #    else:
            #        self.Reconstruction_back = nn.L1Loss().to(self.device)
            if self.train_opt['pixel_criterion']:
                self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])
                self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])
            else:
                self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
                self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # feature loss
            #if self.train_opt['feature_criterion'] == 'l2':
            #    self.Reconstructionf = nn.MSELoss().to(self.device)
            #else:
            #    self.Reconstructionf = nn.L1Loss().to(self.device)
            self.Reconstructionf = ReconstructionLoss(losstype=self.train_opt['feature_criterion'])

            if train_opt['feature_weight'] > 0:
                self.l_fea_w = train_opt['feature_weight']
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)
            else:
                self.l_fea_w = 0

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0


            # optimizers
            # G
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
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], weight_decay=wd_D, betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

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
            self.var_H = self.real_H
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def noise_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, :3, :, :], y)

        if self.train_opt['lambda_mle_forw']:
            z = out[:, 3:, :, :].reshape([out.shape[0], -1])
            l_forw_mle = self.train_opt['lambda_mle_forw'] * torch.sum(torch.norm(z, p=2, dim=1))
        else:
            l_forw_mle = 0

        return l_forw_fit, l_forw_mle

    def loss_backward(self, x, x_samples):
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        # feature loss
        if self.l_fea_w > 0:
            l_back_fea = self.feature_loss(x, x_samples_image)
        else:
            l_back_fea = torch.tensor(0)

        # GAN loss
        pred_g_fake = self.netD(x_samples_image)
        if self.opt['train']['gan_type'] == 'gan':
            l_back_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        elif self.opt['train']['gan_type'] == 'ragan':
            pred_d_real = self.netD(x).detach()
            l_back_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) + self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2

        return l_back_rec, l_back_fea, l_back_gan

    def feature_loss(self, real, fake):
        real_fea = self.netF(real).detach()
        fake_fea = self.netF(fake)
        l_g_fea = self.l_fea_w * self.Reconstructionf(real_fea, fake_fea)
        
        return l_g_fea
        

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.input = self.real_H

        self.output = self.netG(x=self.input)
        loss = 0

        zshape = self.output[:, 3:, :, :].shape

        yy = torch.cat((self.output[:, :3, :, :], self.noise_batch(zshape)), dim=1)

        self.fake_H = self.netG(x=yy, rev=True)

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_forw_fit, l_forw_mle = self.loss_forward(self.output, self.var_L)

            l_back_rec, l_back_fea, l_back_gan = self.loss_backward(self.real_H, self.fake_H)

            loss += l_forw_fit + l_forw_mle + l_back_rec + l_back_fea + l_back_gan

            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())
        if self.opt['train']['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_back_rec'] = l_back_rec.item()
            self.log_dict['l_back_fea'] = l_back_fea.item()
            self.log_dict['l_back_gan'] = l_back_gan.item()
        self.log_dict['l_d'] = l_d_total.item()

    def test(self):
        Lshape = self.var_L.shape
        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        noise_scale = 1

        if self.test_opt and self.test_opt['noise_scale']:
            noise_scale = self.test_opt['noise_scale']

        self.netG.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]

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
        load_path_G2 = self.opt['path']['pretrain_model_G2']
        interpolation = self.opt['interpolation']
        if load_path_G is not None:
            if load_path_G2 == None:
                logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            else:
                logger.info('Loading interpolation model for G [{:s}, {:s}, interpolation:{:s}] ...'.format(load_path_G, load_path_G2, str(interpolation)))
                self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'], load_path2=load_path_G2, interpolation=interpolation)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
