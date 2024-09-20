import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave


def search_LR(img, threshold=35, is_plot=False):
    center_y = int(img.shape[0]/2)
    center_line = img[center_y]
    mean_RGB = center_line.mean(axis=1)
    length = len(mean_RGB)

    indices = np.argwhere(mean_RGB > threshold)
    idx1, idx2 = indices[0][0], indices[-1][0]
    left = mean_RGB[idx1:idx1+1000].argmin()+idx1
    right = mean_RGB[idx2-1000:idx2].argmin()+idx2-1000

    return left, right


def photo_cut(img, name, folder, left, right):
    center = (left+right)//2
    width = 3520
    new_left, new_right = center-width//2, center+width//2
    
    cut_idx = np.arange(new_left, new_right, 1)
    img = np.take(img, cut_idx, axis=1)
    imsave(folder+'{}_ORIG.tif'.format(name[:-4]), img[:3200])


parser = argparse.ArgumentParser(description='Cut old photos -> 3200 * 3520')
parser.add_argument('--name', '-n', default=None, type=str, help='Plot the old photo to find the boundries (left and right) manually')
parser.add_argument('--left', '-l', default=None, type=int, help='The position of the left boundry (int)')
parser.add_argument('--right', '-r', default=None, type=int, help='The position of the left boundry (int)')
args = parser.parse_args()

name = args.name
left = args.left
right = args.right

old_dir = './old/'
new_dir = './photo_cut/'

if name is None:
    threshold = 35

    old_files = [file for file in os.listdir(old_dir) if file.endswith('.tif')]
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    for old_file in old_files:
        img = plt.imread(old_dir+old_file)
        left, right = search_LR(img, threshold=threshold, is_plot=False)
        photo_cut(img, old_file, new_dir, left, right)
        print('{} completed!'.format(old_file))

else:
    img = plt.imread(old_dir+name)
    if (left is None and right is None):
        plt.imshow(img)
        plt.show()
    else:
        photo_cut(img, name, new_dir, left, right)
        print('{} updated!'.format(name))

assert len(os.listdir(old_dir)) == len(os.listdir(new_dir)), \
'The number of files in new_dir {} is not equal to the number of files in old_dir {}! Please check the folders!'.format(len(os.listdir(new_dir)), len(os.listdir(old_dir)))
