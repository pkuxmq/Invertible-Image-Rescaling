import torch
import torch.nn as nn
import numpy as np
import cv2

class JPG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, quality):
        output = input
        batch_size = input.shape[0]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        for i in range(batch_size):
            tensor = input[i, :, :, :].squeeze().float().cpu().clamp_(0, 1)
            img = tensor.numpy()
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
            img = (img * 255.).round().astype(np.uint8)

            _, encimg = cv2.imencode('.jpg', img, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            decimg = decimg * 1.0 / 255.
            decimg = decimg[:, :, [2, 1, 0]]
            dectensor = torch.from_numpy(np.ascontiguousarray(np.transpose(decimg, (2, 0, 1)))).float()
            output[i, :, :, :] = dectensor.cuda()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class apply_jpg(nn.Module):
    def __init__(self):
        super(apply_jpg, self).__init__()

    def forward(self, input, quality):
        return JPG.apply(input, quality)
