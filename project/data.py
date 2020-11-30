"""Data loader."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 30日 星期一 22:50:05 CST
# ***
# ************************************************************************************/
#

import os
import math
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils

# xxxx--modify here
train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"

def multiple_crop(data, multiple=32):
    # Crop image to a multiple
    C, H, W = data.shape
    Hnew = int(H/multiple)*multiple
    Wnew = int(W/multiple)*multiple
    h = (H - Hnew)//2
    w = (W - Wnew)//2
    return data[:, h:h+Hnew, w:w+Wnew]


def multiple_scale(data, multiple=32):
    # Scale image to a multiple
    C, H, W = data.shape
    Hnew = int(multiple * math.ceil(H/multiple))
    Wnew = int(multiple * math.ceil(W/multiple))
    temp = data.new_zeros(C, Hnew, Wnew)
    temp[:, 0:H, 0:W] = data
    return temp


def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)

class ImageZoomBDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms=get_transform()):
        """Init dataset."""
        super(ImageZoomBDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        # xxxx--modify here
        self.images = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        """Load images."""
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = ImageZoomBDataset(train_dataset_rootdir, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    # xxxx--modify here
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl

def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = ImageZoomBDataset(test_dataset_rootdir, get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)

def ImageZoomBDatasetTest():
    """Test dataset ..."""

    ds = ImageZoomBDataset(train_dataset_rootdir)
    print(ds)
    # src, tgt = ds[10]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()

if __name__ == '__main__':
    ImageZoomBDatasetTest()
