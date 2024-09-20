import torch
import torch.nn as nn


def ConvBNReLU(i, o, kernel_size=(3, 3), stride=1, padding=1, groups=1, bn=True, relu=True):
    layers = [nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride,
                        padding=padding, groups=groups, bias=not bn)]

    if bn:
        layers += [nn.BatchNorm2d(o)]

    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


def DeConvBNReLU(i, o, kernel_size=(3, 3), stride=1, padding=1, groups=1, bn=True, relu=True):
    layers = [nn.ConvTranspose2d(i, o, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups, bias=not bn)]

    if bn:
        layers += [nn.BatchNorm2d(o)]

    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
