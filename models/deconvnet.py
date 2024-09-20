import torch
import torch.nn as nn

from models.utils import *


class DeConvNet(nn.Module):
    def __init__(self):
        super(DeConvNet, self).__init__()
        self.conv_1_1 = ConvBNReLU(3, 64)
        self.conv_1_2 = ConvBNReLU(64, 64)
        self.pool_1 = nn.MaxPool2d(2, return_indices=True)

        self.conv_2_1 = ConvBNReLU(64, 128)
        self.conv_2_2 = ConvBNReLU(128, 128)
        self.pool_2 = nn.MaxPool2d(2, return_indices=True)

        self.conv_3_1 = ConvBNReLU(128, 256)
        self.conv_3_2 = ConvBNReLU(256, 256)
        self.pool_3 = nn.MaxPool2d(2, return_indices=True)

        self.conv_4_1 = ConvBNReLU(256, 512)
        self.conv_4_2 = ConvBNReLU(512, 512)
        self.pool_4 = nn.MaxPool2d(2, return_indices=True)

        self.conv_5_1 = ConvBNReLU(512, 1024)
        self.conv_5_2 = ConvBNReLU(1024, 1024)
        self.deconv_5_3 = DeConvBNReLU(1024, 512)
        self.unpool_4 = nn.MaxUnpool2d(2)

        self.deconv_6_1 = DeConvBNReLU(512, 512)
        self.deconv_6_2 = DeConvBNReLU(512, 256)
        self.unpool_3 = nn.MaxUnpool2d(2)

        self.deconv_7_1 = DeConvBNReLU(256, 256)
        self.deconv_7_2 = DeConvBNReLU(256, 128)
        self.unpool_2 = nn.MaxUnpool2d(2)

        self.deconv_8_1 = DeConvBNReLU(128, 128)
        self.deconv_8_2 = DeConvBNReLU(128, 64)
        self.unpool_1 = nn.MaxUnpool2d(2)

        self.deconv_9_1 = DeConvBNReLU(64, 64)
        self.deconv_9_2 = DeConvBNReLU(64, 64)
        self.conv_1x1 = nn.Conv2d(64, 2, (1, 1))

    def forward(self, x):
        x, index_1 = self.pool_1(self.conv_1_2(self.conv_1_1(x)))
        x, index_2 = self.pool_2(self.conv_2_2(self.conv_2_1(x)))
        x, index_3 = self.pool_3(self.conv_3_2(self.conv_3_1(x)))
        x, index_4 = self.pool_4(self.conv_4_2(self.conv_4_1(x)))

        x = self.conv_5_2(self.conv_5_1(x))
        x = self.deconv_5_3(x)
        x = self.unpool_4(x, index_4)

        x = self.unpool_3(self.deconv_6_2(self.deconv_6_1(x)), index_3)
        x = self.unpool_2(self.deconv_7_2(self.deconv_7_1(x)), index_2)
        x = self.unpool_1(self.deconv_8_2(self.deconv_8_1(x)), index_1)

        x = self.deconv_9_2(self.deconv_9_1(x))
        out = self.conv_1x1(x)
        return out
