import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


class CellImages(Dataset):
    def __init__(self, data_dir, indices, img_transform=None, joint_transform=None):
        self.data_dir = data_dir
        self.imgs = [idx+'_ORIG.tif' for idx in indices if os.path.exists(os.path.join(data_dir, idx+'_ORIG.tif'))]
        self.masks = [idx+'_PS.tif' for idx in indices if os.path.exists(os.path.join(data_dir, idx+'_PS.tif'))]
        if len(self.imgs) == 0 or len(self.masks) == 0:
            raise RuntimeError('Found 0 images (masks), please check the directory.')
        if len(self.imgs) != len(self.masks):
            raise RuntimeError('The number of images is not equal to the number of masks, please check the dataset.')
        self.img_transform = img_transform
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file, mask_file = self.imgs[index], self.masks[index]
        img = plt.imread(os.path.join(self.data_dir, img_file))
        mask = plt.imread(os.path.join(self.data_dir, mask_file))
        img, mask = img[:320, :320], mask[:320, :320]

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
            mask = np.asarray(mask)
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Mask pixel values: cell=1, background=0
        mask = np.min(mask, axis=2)
        mask = np.where(mask < 250, 1, 0)
        mask = torch.from_numpy(mask)

        return img, mask
