"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 30日 星期一 22:50:05 CST
# ***
# ************************************************************************************/
#

import functools
import math
import os
import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
# from torch.utils.checkpoint import checkpoint_sequential
from tqdm import tqdm
from model_helper import ImageZoomModel
from data import gaussian_batch

def PSNR(img1, img2):
    """PSNR."""
    difference = (1.*img1-img2)**2
    mse = torch.sqrt(torch.mean(difference)) + 0.000001
    return 20*torch.log10(1./mse)

def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)

def export_onnx_model():
    """Export onnx model."""

    import onnx
    from onnx import optimizer

    onnx_file = "models/image_zoom.onnx"
    weight_file = "models/ImageZoom.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)

    input_names = ["input"]
    output_names = ["output"]
    # variable lenght axes
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('image_clean.onnx')"


def export_torch_model():
    """Export torch model."""

    script_file = "models/image_zoom.pt"
    weight_file = "models/ImageZoom.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


def get_model(scale=4):
    """Create model."""
    model_setenv()
    model = ImageZoomModel(channel_in=3, channel_out=3, scale=scale)
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x, target):
        return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))

class L1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1Loss, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        diff = x - target
        return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))


# def loss_forward(output, y, z, scale=4.0):
#     # output, y, z = output[:, :3, :, :], LR, output[:, 3:, :, :]
#     l_forw_fit = (scale ** 2) * L2Loss()(output, y)
#     z = z.reshape([output.shape[0], -1])
#     l_forw_ce = 1.0 * torch.sum(z**2) / z.shape[0]
#     return l_forw_fit, l_forw_ce

# def loss_backward(model, x, y):
#     x_samples = model(x=y, rev=True)
#     # (Pdb) x_samples.size()
#     # torch.Size([8, 3, 246, 256])
#     l_back_rec = 1.0 * L1Loss()(x, x_samples)
#     return l_back_rec

def train_epoch(loader, model, optimizer, device, scale, tag=''):
    """Trainning model ..."""

    total_loss = Counter()
    model.train()

    forward_loss = L2Loss()
    backward_loss = L1Loss()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            LR, HR = data
            count = len(LR)

            # Transform data to device
            LR = LR.to(device)
            HR = HR.to(device)

            output = model(x=HR)
            zshape = output[:, 3:, :, :].shape
            # z -- output[:, 3:, :, :]
            # torch.Size([2, 48, 64, 64])
            # torch.Size([2, 45, 64, 64])
            l_forw_fit = (scale ** 2) * forward_loss(output[:, :3, :, :], LR)
            z = output[:, 3:, :, :].reshape([output.shape[0], -1])
            l_forw_ce = 1.0 * torch.sum(z**2) / z.shape[0]

            y_ = torch.cat((output[:, :3, :, :], gaussian_batch(zshape).to(device)), dim=1)
            x_samples = model(x=y_, rev=True)
            # (Pdb) x_samples.size()
            # torch.Size([8, 3, 246, 256])
            l_back_rec = 1.0 * backward_loss(HR, x_samples)

            # total loss
            loss = l_forw_fit + l_forw_ce + l_back_rec

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print("l_forw_fit is {}, l_forw_ce is {}, l_back_rec is {}".format( \
                    l_forw_fit, l_forw_ce, l_back_rec))
                print("LR.size is {}".format(LR.size()))
                print("HR.size is {}".format(HR.size()))
                print("output.size is {}".format(output.size()))
                print("output.mean is {}, output.min is {}, output.max is {}".format(
                    output.mean(), output.min(), output.max()))
                print("y_.size is {}".format(y_.size()))
                print("y_.mean is {}, y_.min is {}, y_.max is {}".format(
                    y_.mean(), y_.min(), y_.max()))
                print("x_samples.size is {}".format(x_samples.size()))
                print("x_samples.mean is {}, x_samples.min is {}, x_samples.max is {}".format(
                    x_samples.mean(), x_samples.min(), x_samples.max()))
                pdb.set_trace()

                sys.exit(1)
            # Update loss
            total_loss.update(loss_value, count)

            del output, l_forw_fit, l_forw_ce, l_back_rec, y_, LR, HR, z
            torch.cuda.empty_cache()

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            # Lipschit condition !!!
            # nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, scale, tag=''):
    """Validating model  ..."""

    valid_lr_loss = Counter()
    valid_hr_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            LR, HR = data
            count = len(LR)

            # Transform data to device
            LR = LR.to(device)
            HR = HR.to(device)

            # Predict results without calculating gradients
            B,C,H,W = LR.size()
            zshape = [B, C * (scale**2 - 1), H, W]
            with torch.no_grad():
                forw_L = model(x=HR)[:, :3, :, :]
                y_forw = torch.cat((forw_L, gaussian_batch(zshape).to(device)), dim=1)
                predicts = model(x=y_forw, rev=True)[:, :3, :, :]

            psnr_lr = PSNR(forw_L, LR)
            psnr_hr = PSNR(HR, predicts)

            valid_lr_loss.update(psnr_lr.item(), count)
            valid_hr_loss.update(psnr_hr.item(), count)
            t.set_postfix(loss='LR PSNR: {:.6f}, HR PSNR: {:.6f}'.format(valid_lr_loss.avg, valid_hr_loss.avg))
            t.update(count)

            del LR, HR, forw_L, y_forw, predicts, psnr_lr, psnr_hr
            torch.cuda.empty_cache()

def model_device():
    """First call model_setenv. """
    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

   # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        os.environ["ENABLE_APEX"] = "YES"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def enable_amp(x):
    """Init Automatic Mixed Precision(AMP)."""
    if os.environ["ENABLE_APEX"] == "YES":
        x = amp.initialize(x, opt_level="O1")


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = model_device()

    model = ImageZoomBModel()
    model.eval()
    model = model.to(device)
    enable_amp(model)

    progress_bar = tqdm(total=100)
    progress_bar.set_description("Test Inference Performance ...")

    for i in range(100):
        # xxxx--modify here
        input = torch.randn(64, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        progress_bar.update(1)


if __name__ == '__main__':
    """Test model ..."""

    model = get_model()
    print(model)

    # export_torch_model()
    # export_onnx_model()

    # infer_perform()
