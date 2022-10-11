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
from models.modules.Replace import Replace

logger = logging.getLogger('base')

class IRNCRMModel(BaseModel):
    def __init__(self, opt):
        super(IRNCRMModel, self).__init__(opt)

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

        self.netR = networks.define_R(opt).to(self.device)
        if opt['dist']:
            self.netR = DistributedDataParallel(self.netR, device_ids=[torch.cuda.current_device()])
        else:
            self.netR = DataParallel(self.netR)

        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()
        self.apply_jpg = apply_jpg()
        self.Replace = Replace()

        if self.is_train:
            self.netG.train()
            self.netR.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            self.Reconstruction_jpeg = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_jpeg'])


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            wd_R = train_opt['weight_decay_R'] if train_opt['weight_decay_R'] else 0
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

            optim_params = []
            for k, v in self.netR.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_R = torch.optim.Adam(optim_params, lr=train_opt['lr_R'],
                                                weight_decay=wd_R,
                                                betas=(train_opt['beta1_R'], train_opt['beta2_R']))
            self.optimizers.append(self.optimizer_R)

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

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = z.reshape([out.shape[0], -1])
        l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def loss_backward(self, x, y):
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)

        return l_back_rec


    def optimize_parameters(self, step):
        if self.train_opt['only_jpeg_reconstruction']:
            for p in self.netG.parameters():
                p.requires_grad = False

            self.optimizer_R.zero_grad()

            self.input = self.real_H
            with torch.no_grad():
                self.output = self.netG(x=self.input)
                LR = self.Quantization(self.output[:, :3, :, :])
                LR_ = LR.clone()
                quality = self.train_opt['jpg_quality']
                self.output_jpeg = self.apply_jpg(LR_, quality).detach()

            self.output_restore = self.netR(self.output_jpeg)
            l_jpeg_rec = self.train_opt['lambda_rec_jpeg'] * self.Reconstruction_jpeg(LR, self.output_restore)
            loss = l_jpeg_rec

            if self.train_opt['add_joint_loss']:
                start_iter = self.train_opt['joint_loss_iters'] if self.train_opt['joint_loss_iters'] != None else -1
                if step > start_iter:
                    self.output_restore = self.Quantization(self.output_restore)
                    zshape = self.output[:, 3:, :, :].shape
                    gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
                    y_ = torch.cat((self.output_restore, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
                    x_samples = self.netG(x=y_, rev=True)[:, :3, :, :]
                    l_back_rec = self.train_opt['lambda_joint_back'] * self.Reconstruction_back(self.real_H, x_samples)
                    loss += l_back_rec
            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netR.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_R.step()

            # set log
            self.log_dict['l_jpeg_rec'] = l_jpeg_rec.item()
            if self.train_opt['add_joint_loss'] and step > start_iter:
                self.log_dict['l_back_rec'] = l_back_rec.item()

            for p in self.netG.parameters():
                p.requires_grad = True

        else:
            self.optimizer_G.zero_grad()
            self.optimizer_R.zero_grad()

            # forward downscaling
            self.input = self.real_H
            self.output = self.netG(x=self.input)

            zshape = self.output[:, 3:, :, :].shape
            LR_ref = self.ref_L.detach()

            l_forw_fit, l_forw_ce = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

            # backward upscaling
            LR = self.Quantization(self.output[:, :3, :, :])
            LR_ = LR.clone()
            quality = self.train_opt['jpg_quality']
            self.output_jpeg = self.apply_jpg(LR_, quality).detach()
            self.output_restore = self.netR(x=self.output_jpeg)
            l_jpeg_rec = self.train_opt['lambda_rec_jpeg'] * self.Reconstruction_jpeg(LR, self.output_restore)

            LR = self.Replace(LR, self.Quantization(self.output_restore))
            gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
            y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

            l_back_rec = self.loss_backward(self.real_H, y_)

            # total loss
            loss = l_jpeg_rec + l_forw_fit + l_back_rec + l_forw_ce
            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
                nn.utils.clip_grad_norm_(self.netR.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_G.step()
            self.optimizer_R.step()

            # set log
            self.log_dict['l_forw_fit'] = l_forw_fit.item()
            self.log_dict['l_forw_ce'] = l_forw_ce.item()
            self.log_dict['l_back_rec'] = l_back_rec.item()
            self.log_dict['l_jpeg_rec'] = l_jpeg_rec.item()

    def test(self):
        if self.test_opt and self.test_opt['bic_crm']:
            self.netR.eval()
            quality = self.test_opt['jpg_quality']
            with torch.no_grad():
                self.jpeg_L = self.apply_jpg(self.ref_L, quality)
                self.restore_L = self.netR(x=self.jpeg_L)
                self.restore_L = self.Quantization(self.restore_L)
            self.forw_L = self.ref_L
            self.fake_H = self.restore_L
            self.netR.train()
            return
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        self.netR.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)
            forw_L_ = self.forw_L.clone()
            quality = self.test_opt['jpg_quality']
            self.jpeg_L = self.apply_jpg(forw_L_, quality).detach()
            if self.test_opt['ignore_restore']:
                self.restore_L = self.jpeg_L
            else:
                self.restore_L = self.netR(x=self.jpeg_L)
                self.restore_L = self.Quantization(self.restore_L)

            y_forw = torch.cat((self.restore_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]


        self.netG.train()
        self.netR.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(LR_img)
        self.netG.train()

        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
        self.netG.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['RLR'] = self.restore_L.detach()[0].float().cpu()
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
            
        s, n = self.get_network_description(self.netR)
        if isinstance(self.netR, nn.DataParallel) or isinstance(self.netR, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netR.__class__.__name__,
                                             self.netR.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netR.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

        load_path_R = self.opt['path']['pretrain_model_R']
        if load_path_R is not None:
            logger.info('Loading model for R [{:s}] ...'.format(load_path_R))
            self.load_network(load_path_R, self.netR, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netR, 'R', iter_label)
