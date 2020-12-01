"""Model predict."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 30日 星期一 22:50:05 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from data import gaussian_batch
from model import enable_amp, get_model, model_device, model_load, PSNR

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/ImageZoom_X4.pth", help="checkpint file")
    parser.add_argument('--input', type=str,
                        default="dataset/predict/LR/*.png", help="input image")
    parser.add_argument('--scale', type=int, default=4, help="scale factor")
    parser.add_argument('--output', type=str,
                        default="dataset/predict/HR", help="output directory")

    args = parser.parse_args()

    if (args.scale == 2):
        args.checkpoint = "models/ImageZoom_X2.pth"

    model = get_model(scale=args.scale)
    model_load(model, args.checkpoint)
    device = model_device()
    model.to(device)
    model.eval()

    enable_amp(model)

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        B,C,H,W = input_tensor.size()
        zshape = [B, C * (args.scale**2 - 1), H, W]

        best_psnr = 0;
        for i in range(10):
            with torch.no_grad():
                y_forw = torch.cat((input_tensor, gaussian_batch(zshape).to(device)), dim=1)
                output_tensor = model(x=y_forw, rev=True)[:, :3, :, :]
                LR_fake = model(x=output_tensor)[:, :3, :, :]
                psnr = PSNR(input_tensor, LR_fake).item()
                if (psnr > best_psnr):
                    best_psnr = psnr
                    final_output_tensor = output_tensor.cpu()
        final_output_tensor.clamp_(0, 1.0)
        final_output_tensor = final_output_tensor.squeeze(0)

        toimage(final_output_tensor.cpu()).save(
            "{}/{}".format(args.output, os.path.basename(filename)))

        del input_tensor, output_tensor, final_output_tensor
        torch.cuda.empty_cache()

        progress_bar.set_postfix(PSNR='{:.2f}'.format(best_psnr))
