# Visualization of the features of the first convolutional layer

import numpy as np
import matplotlib.pyplot as plt

import torch
from models import *

checkpoint = torch.load('./checkpoint/training_saved.t7', map_location='cpu')
net = DeConvNet()
net.load_state_dict(checkpoint['net'])

weights = net.conv_1_1[0]._parameters['weight'].data.numpy()

features = []
for weight in weights:
    img = weight.copy()
    img -= img.mean()
    img /= abs(img).max()
    img += 255/2
    img *= 255/2
    img = np.transpose(img, (1, 2, 0))
    features.append(img.astype(np.uint8))

fig = plt.figure(figsize=(8, 8))
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(features[i])
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
