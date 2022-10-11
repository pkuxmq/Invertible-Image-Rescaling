import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    test_results['psnr_grey'] = []
    test_results['ssim_grey'] = []

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals()

        color_img = util.tensor2img(visuals['Color'])  # uint8
        gt_img = util.tensor2img(visuals['GT'])  # uint8
        grey_img = util.tensor2img(visuals['Grey'])  # uint8
        greygt_img = util.tensor2img(visuals['Grey_ref'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(color_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_GT.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_GT.png')
        util.save_img(gt_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_Grey.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_Grey.png')
        util.save_img(grey_img, save_img_path)

        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '_Grey_ref.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '_Grey_ref.png')
        util.save_img(greygt_img, save_img_path)

        # calculate PSNR and SSIM
        gt_img = gt_img / 255.
        color_img = color_img / 255.

        grey_img = grey_img / 255.
        greygt_img = greygt_img / 255.

        crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
        if crop_border == 0:
            cropped_color_img = color_img
            cropped_gt_img = gt_img
        else:
            cropped_color_img = color_img[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

        psnr = util.calculate_psnr(cropped_color_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_color_img * 255, cropped_gt_img * 255)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        # PSNR and SSIM for grey
        psnr_grey = util.calculate_psnr(grey_img * 255, greygt_img * 255)
        ssim_grey = util.calculate_ssim(grey_img * 255, greygt_img * 255)
        test_results['psnr_grey'].append(psnr_grey)
        test_results['ssim_grey'].append(ssim_grey)

        logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}. Grey PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim, psnr_grey, ssim_grey))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    ave_psnr_grey = sum(test_results['psnr_grey']) / len(test_results['psnr_grey'])
    ave_ssim_grey = sum(test_results['ssim_grey']) / len(test_results['ssim_grey'])

    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. Grey psnr: {:.6f} db; ssim: {:.6f}.\n'.format(
            test_set_name, ave_psnr, ave_ssim, ave_psnr_grey, ave_ssim_grey))
