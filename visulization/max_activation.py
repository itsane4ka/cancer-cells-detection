import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from models import *


checkpoint = torch.load('./checkpoint/training_saved.t7', map_location='cpu')
net = DeConvNet()
net.load_state_dict(checkpoint['net'])
net.eval()

file = './input.tif'  # Speicify the input image
img = plt.imread(file)

fig = plt.figure()
ax = fig.add_subplot(1, 6, 1)
ax.imshow(img)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
])

img = img_transform(img)
x = img.reshape(1, 3, 320, 320)


def plot_max_activation(x, fig, position):
    dec = x[0].data.numpy()
    idx = dec.mean(axis=(1, 2)).argmax()
    ax = fig.add_subplot(1, 6, position)
    ax.imshow(dec[idx], cmap='plasma')


x, index_1 = net.pool_1(net.conv_1_2(net.conv_1_1(x)))
x, index_2 = net.pool_2(net.conv_2_2(net.conv_2_1(x)))
x, index_3 = net.pool_3(net.conv_3_2(net.conv_3_1(x)))
x, index_4 = net.pool_4(net.conv_4_2(net.conv_4_1(x)))
x = net.conv_5_2(net.conv_5_1(x))

x = net.deconv_5_3(x)
plot_max_activation(x, fig, 2)

x = net.unpool_4(x, index_4)
x = net.deconv_6_2(net.deconv_6_1(x))
plot_max_activation(x, fig, 3)

x = net.unpool_3(x, index_3)
x = net.deconv_7_2(net.deconv_7_1(x))
plot_max_activation(x, fig, 4)

x = net.unpool_2(x, index_2)
x = net.deconv_8_2(net.deconv_8_1(x))
plot_max_activation(x, fig, 5)

x = net.unpool_1(x, index_1)
x = net.deconv_9_2(net.deconv_9_1(x))
plot_max_activation(x, fig, 6)

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
