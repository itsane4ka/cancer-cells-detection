import torch
import torch.nn as nn


def EncoderBlock(i, o, kernel_size=(3, 3), stride=1, padding=1, bn=True):
    layers = [nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(o)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(o, o, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(o)]
    layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


def DecoderBlock(i, o, kernel_size=(3, 3), stride=1, padding=1, bn=True):
    layers = [nn.Conv2d(i, o*2, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(o*2)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(o*2, o*2, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=not bn)]
    if bn:
        layers += [nn.BatchNorm2d(o*2)]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.ConvTranspose2d(o*2, o, kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc_1 = EncoderBlock(3, 64)
        self.pool_1 = nn.MaxPool2d(2)
        self.enc_2 = EncoderBlock(64, 128)
        self.pool_2 = nn.MaxPool2d(2)
        self.enc_3 = EncoderBlock(128, 256)
        self.pool_3 = nn.MaxPool2d(2)
        self.enc_4 = EncoderBlock(256, 512)
        self.pool_4 = nn.MaxPool2d(2)

        self.dec_4 = DecoderBlock(512, 512)
        self.dec_3 = DecoderBlock(1024, 256)
        self.dec_2 = DecoderBlock(512, 128)
        self.dec_1 = DecoderBlock(256, 64)
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(1, 1)),
        )

    def forward(self, x):
        enc1 = self.enc_1(x)
        enc2 = self.enc_2(self.pool_1(enc1))
        enc3 = self.enc_3(self.pool_2(enc2))
        enc4 = self.enc_4(self.pool_3(enc3))
        dec4 = self.dec_4(self.pool_4(enc4))
        dec3 = self.dec_3(torch.cat((dec4, enc4), dim=1))
        dec2 = self.dec_2(torch.cat((dec3, enc3), dim=1))
        dec1 = self.dec_1(torch.cat((dec2, enc2), dim=1))
        out = self.final(torch.cat((dec1, enc1), dim=1))
        return out
