import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import *

logger = logging.getLogger('base')

class Config_MMD(object):
    def __init__(self, device, mmd_kernels):
        self.device = device
        self.mmd_kernels = mmd_kernels

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

        if self.is_train:
            self.netG.train()

            # loss
            mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
            mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
            #config_mmd_forw = Config_MMD(self.device, train_opt['mmd_forw_kernels'])
            #config_mmd_back = Config_MMD(self.device, train_opt['mmd_back_kernels'])
            config_mmd_forw = Config_MMD(self.device, mmd_forw_kernels)
            config_mmd_back = Config_MMD(self.device, mmd_back_kernels)

            self.MMD_forward = MMDLoss(config_mmd_forw) 
            self.MMD_backward = MMDLoss(config_mmd_back) 

            if self.train_opt['pixel_criterion']:
                self.Reconstruction = ReconstructionLoss(losstype=self.train_opt['pixel_criterion'])
            else:
                self.Reconstruction = ReconstructionLoss()


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                #else:
                #    if self.rank <= 0:
                #        logger.waring('Params [{:s}] will not optimize.'.format(k))
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

    def zeros_batch(self, dims):
        return torch.zeros(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        output_block_grad = torch.cat((out[:, :3, :, :], out[:, 3:, :, :].data), dim=1)
        y_short = torch.cat((y[:, :3, :, :], y[:, 3:, :, :]), dim=1)

        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction(out[:, :3, :, :], y[:, :3, :, :])
        l_forw_mmd = self.train_opt['lambda_mmd_forw'] * torch.mean(self.MMD_forward(output_block_grad, y_short))

        z = out[:, 3:, :, :].reshape([out.shape[0], -1])
        l_forw_mle = self.train_opt['lambda_mle_forw'] * torch.sum(torch.norm(z, p=2, dim=1))

        #print(l_forw_fit.item(), l_forw_mmd.item(), l_forw_mle.item())

        return l_forw_fit, l_forw_mmd, l_forw_mle

    def loss_backward(self, x, y):
        x_samples = self.netG(y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        MMD = self.MMD_backward(x, x_samples_image)
        l_back_mmd = self.train_opt['lambda_mmd_back'] * torch.mean(MMD)
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction(x, x_samples_image)
        if self.train_opt['padding_x']:
            xx = x_samples[:, 3:, :, :].reshape([x_samples.shape[0], -1])
            l_back_mle = self.train_opt['lambda_mle_back'] * torch.sum(torch.norm(xx, p=2, dim=1))
        else:
            l_back_mle = 0

        #print(l_back_mmd.item(), l_back_rec.item())

        return l_back_mmd, l_back_rec, l_back_mle

    #def loss_backward_rec(self, out_y, y, x):
        

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        if self.train_opt['padding_x']:
            self.padding_x_dim = self.train_opt['padding_x']
            padding_xshape = [self.real_H.shape[0], self.padding_x_dim, self.real_H.shape[2], self.real_H.shape[3]]
            self.input = torch.cat((self.real_H, self.noise_batch(padding_xshape)), dim=1)
        else:
            self.input = self.real_H

        if self.train_opt['use_jacobian']:
            self.output, jacobian = self.netG(self.input, rev=False, cal_jacobian=True)
            loss = -jacobian
        else:
            self.output = self.netG(self.input)
            loss = 0
            
        zshape = self.output[:, 3:, :, :].shape
        y = torch.cat((self.var_L, self.noise_batch(zshape)), dim=1)
        #loss = self.loss_forward(self.output, y) + self.loss_backward(self.real_H, y)

        if self.train_opt['use_learned_y']:
            yy = torch.cat((self.output[:, :3, :, :], self.noise_batch(zshape)), dim=1)
        else:
            yy = y

        l_forw_fit, l_forw_mmd, l_forw_mle = self.loss_forward(self.output, y)

        if (self.train_opt['lambda_rec_back'] == 0.0 and self.train_opt['not_use_back_mmd']):
            l_back_mmd, l_back_rec, l_back_mle = 0, 0
        else:
            l_back_mmd, l_back_rec, l_back_mle = self.loss_backward(self.real_H, yy)
            if self.train_opt['learned_y_par']:
                _, l_back_rec_bicubic, l_back_mle_bicubic = self.loss_backward(self.real_H, y)
                l_back_rec = self.train_opt['learned_y_par'] * l_back_rec + (1 - self.train_opt['learned_y_par']) * l_back_rec_bicubic
                l_back_mle = self.train_opt['learned_y_par'] * l_back_mle + (1 - self.train_opt['learned_y_par']) * l_back_mle_bicubic

        if self.train_opt['use_stage']:
            if step < self.train_opt['stage1_step']:
                l_back_rec = 0
                l_back_mle = 0

        loss += l_forw_fit + l_back_rec + l_forw_mle + l_back_mle

        if not self.train_opt['not_use_forw_mmd']:
            loss += l_forw_mmd

        if not self.train_opt['not_use_back_mmd']:
            loss += l_back_mmd

        #print(loss.item())

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        ## set log
        #self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        Lshape = self.var_L.shape
        if self.train_opt and self.train_opt['padding_x']:
            self.padding_x_dim = self.train_opt['padding_x']
            padding_xshape = [self.real_H.shape[0], self.padding_x_dim, self.real_H.shape[2], self.real_H.shape[3]]
            self.input = torch.cat((self.real_H, self.noise_batch(padding_xshape)), dim=1)
            input_dim = self.padding_x_dim + Lshape[1]
        else:
            input_dim = Lshape[1]
            self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        noise_scale = 1

        if self.test_opt and self.test_opt['noise_scale']:
            noise_scale = self.test_opt['noise_scale']
        y = torch.cat((self.var_L, noise_scale * self.noise_batch(zshape)), dim=1)

        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(y, rev=True)[:, :3, :, :]
            
            self.forw_L = self.netG(self.input)[:, :3, :, :]

        y_forw = torch.cat((self.forw_L, noise_scale * self.noise_batch(zshape)), dim=1)
        with torch.no_grad():
            self.fake_H_forw = self.netG(y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    #def test_x8(self):
    #    # from https://github.com/thstkdgus35/EDSR-PyTorch
    #    self.netG.eval()

    #    def _transform(v, op):
    #        # if self.precision != 'single': v = v.float()
    #        v2np = v.data.cpu().numpy()
    #        if op == 'v':
    #            tfnp = v2np[:, :, :, ::-1].copy()
    #        elif op == 'h':
    #            tfnp = v2np[:, :, ::-1, :].copy()
    #        elif op == 't':
    #            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    #        ret = torch.Tensor(tfnp).to(self.device)
    #        # if self.precision == 'half': ret = ret.half()

    #        return ret

    #    lr_list = [self.var_L]
    #    for tf in 'v', 'h', 't':
    #        lr_list.extend([_transform(t, tf) for t in lr_list])
    #    with torch.no_grad():
    #        sr_list = [self.netG(aug) for aug in lr_list]
    #    for i in range(len(sr_list)):
    #        if i > 3:
    #            sr_list[i] = _transform(sr_list[i], 't')
    #        if i % 4 > 1:
    #            sr_list[i] = _transform(sr_list[i], 'h')
    #        if (i % 4) % 2 == 1:
    #            sr_list[i] = _transform(sr_list[i], 'v')

    #    output_cat = torch.cat(sr_list, dim=0)
    #    self.fake_H = output_cat.mean(dim=0, keepdim=True)
    #    self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['SR_forw'] = self.fake_H_forw.detach()[0].float().cpu()
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
