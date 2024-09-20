import numpy as np
import random

from PIL import Image, ImageOps


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ArrayToPIL(object):
    def __call__(self, img, mask):
        return Image.fromarray(img), Image.fromarray(mask)


class RandomRotateFlip(object):
    def __call__(self, img, mask):
        p_rot = random.random()
        if 0.25 <= p_rot < 0.5:
            img, mask = img.transpose(Image.ROTATE_90), mask.transpose(Image.ROTATE_90)
        elif 0.5 <= p_rot < 0.75:
            img, mask = img.transpose(Image.ROTATE_180), mask.transpose(Image.ROTATE_180)
        elif p_rot >= 0.75:
            img, mask = img.transpose(Image.ROTATE_270), mask.transpose(Image.ROTATE_270)

        p_flip = random.random()
        if p_flip >= 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask
