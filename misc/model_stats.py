import sys
sys.path.append(dir)  # specify the model directory

from models import *

import torch
import torch.nn as nn


def model_stats(net, image_size):
    layer_names = []
    layer_params = []
    layer_activs = []
    layer_multadds = []

    x = torch.rand(1, 3, image_size, image_size)  # input
    print('Input size: {}'.format(tuple(x.size())))
    print('{0:20}{1:>15}{2:>15}{3:>15}'.format('Layer', 'Params', 'Activs', 'Mult-Adds'))
    print('-'*65)
    pool_idx = []

    net.eval()
    with torch.no_grad():
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):  # convolutional layers
                h_input, w_input = x.size()[2:]
                h_kernel, w_kernel = module.kernel_size

                if module.in_channels == 2*x.size()[1]:  # concatenation
                    x = module(torch.cat((x, x), dim=1))
                else:
                    x = module(x)

                layer_names.append(name)
                layer_params.append(module.weight.data.numel())
                layer_activs.append(x.numel())

                if module.groups == 1:
                    layer_multadds.append(h_kernel*w_kernel*module.in_channels*module.out_channels*h_input*w_input)
                else:  # depth-wise convolution
                    layer_multadds.append(h_kernel*w_kernel*module.in_channels*h_input*w_input)

                print('{0:20}{1:15}{2:15}{3:15}'.format(layer_names[-1], layer_params[-1], layer_activs[-1], layer_multadds[-1]))

            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.MaxUnpool2d):  # pooling/unpooling layers
                if isinstance(module, nn.MaxPool2d):
                    out = module(x)
                    if isinstance(out, tuple):  # return_indices
                        x = out[0]
                        pool_idx.append(out[1])
                    else:
                        x = out
                else:
                    indices = pool_idx.pop()
                    x = module(x, indices)

                layer_names.append(name)
                layer_params.append(0)
                layer_activs.append(x.numel())
                layer_multadds.append(0)

                print('{0:20}{1:15}{2:15}{3:15}'.format(layer_names[-1], layer_params[-1], layer_activs[-1], layer_multadds[-1]))

    print('-'*65)
    return layer_names, layer_params, layer_activs, layer_multadds


if __name__ == '__main__':
    # net = UNet()
    # net = DeConvSkipNet()
    net = MobileV2SegNet(width_multiplier=1.)

    image_size = 320
    stats = model_stats(net, image_size)
    print('Total number of parameters:', sum(stats[1]))
    print('Total number of activations:', sum(stats[2])+1*3*image_size**2)
    print('Total number of Mult-Adds:', sum(stats[3]))
