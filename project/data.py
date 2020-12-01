"""Data loader."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:48:28 CST
# ***
# ************************************************************************************/
#
import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image
import pdb

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"


def get_transform(train=True):
    """Transform images."""

    ts = []
    # if train:
    #     # ts.append(T.RandomHorizontalFlip(0.5))
    #     PATH_SIZE=(256, 256)
    #     ts.append(T.RandomCrop(PATH_SIZE))
    ts.append(T.ToTensor())
    return T.Compose(ts)


def random_crop(LR, HR):
    # Patch Size
    PATCH_SIZE = 32
    H, W = LR.shape[1:]
    h = random.randint(0, H - PATCH_SIZE)
    w = random.randint(0, W - PATCH_SIZE)
    return LR[:, h:h+PATCH_SIZE, w:w+PATCH_SIZE], HR[:, 4*h:4*(h+PATCH_SIZE), 4*w:4*(w+PATCH_SIZE)]


class ImageZoomDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, trainning, transforms=get_transform()):
        """Init dataset."""
        super(ImageZoomDataset, self).__init__()
        self.trainning = trainning

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = list(sorted(os.listdir(root + "/LR")))

    def __getitem__(self, idx):
        """Load images."""
        img_path = os.path.join(self.root + "/LR", self.images[idx])
        lr = Image.open(img_path).convert("RGB")

        img_path = os.path.join(self.root + "/HR", self.images[idx])
        hr = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            lr = self.transforms(lr)
            hr = self.transforms(hr)

        if self.trainning:
            lr, hr = random_crop(lr, hr)

        return lr, hr

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
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class ZeroShotImageZoomDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, image_file_name, transforms=get_transform()):
        """Init dataset."""
        super(ZeroShotImageZoomDataset, self).__init__()
        self.image_file_name = image_file_name
        self.transforms = transforms
        image = Image.open(image_file_name).convert("RGB")
        self.hr = self.transforms(image)
        W, H = image.size
        image = image.resize((W//4, H//4), Image.ANTIALIAS)
        self.lr = self.transforms(image)

    def __getitem__(self, idx):
        """Load images."""
        return random_crop(self.lr, self.hr)

    def __len__(self):
        """Return total numbers of images."""
        return 1000

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Image File: {}\n'.format(self.image_file_name)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def zeroshot_data(file_name, bs):
    """Get data loader for trainning & validating, bs means batch_size."""
    ds = ZeroShotImageZoomDataset(file_name, get_transform(train=True))
    # Define training and validation data loaders
    dl = data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    return dl

def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = ImageZoomDataset(
        train_dataset_rootdir, True, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(
        valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = ImageZoomDataset(
        test_dataset_rootdir, False, get_transform(train=False))
    test_dl = data.DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def ImageZoomDatasetTest():
    """Test dataset ..."""

    ds = ImageZoomDataset(train_dataset_rootdir, True)
    print(ds)
    # src, tgt = ds[10]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()


def gaussian_batch(dims):
    return torch.randn(tuple(dims))

if __name__ == '__main__':
    ImageZoomDatasetTest()
