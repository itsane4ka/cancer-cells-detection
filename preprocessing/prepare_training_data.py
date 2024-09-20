# 1. Obtain the microscopic view of the culture dish with cells from the original photo;
# 2. Then crop it to one hundred 320*320 images.
# The code here will be reorganized.

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave


def search_LR(img, threshold, is_plot=False):
    center_y = int(img.shape[0]/2)
    center_line = img[center_y]
    mean_RGB = center_line.mean(axis=1)
    length = len(mean_RGB)

    indices = np.argwhere(mean_RGB > threshold)
    idx1, idx2 = indices[0][0], indices[-1][0]
    left = mean_RGB[idx1:idx1+1000].argmin()+idx1
    right = mean_RGB[idx2-1000:idx2].argmin()+idx2-1000

    if is_plot:
        plt.plot(range(length), mean_RGB, 'b', label='{} - {} = {}'.format(right, left, right-left))
        plt.legend(loc='lower center')
        plt.show()

    return left, right


def tif_cut(img, left, right):
    cut_idx = np.arange(left, right, 1)
    img_v = np.take(img, cut_idx, axis=1)
    return img_v[:3200]


def preprocess(series, img_dir, mask_dir, target_dir, threshold=35):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        print('Target directory created')

    img_files = [file for file in os.listdir(img_dir) if file.endswith('.tif')]
    mask_files = [file for file in os.listdir(mask_dir) if file.endswith('.tif')]
    filenames = set(img_files).intersection(mask_files)
    print('img: {}, mask: {}, matches: {}'.format(len(img_files), len(mask_files), len(filenames)))

    for idx, filename in enumerate(filenames):
        img = plt.imread(os.path.join(img_dir, filename))
        mask = plt.imread(os.path.join(mask_dir, filename))

        left, right = search_LR(img, threshold, is_plot=False)

        center = (left+right)//2
        width = 3520
        left, right = center-width//2, center+width//2

        try:
            img = tif_cut(img, left, right)
            if img.shape != (3200, 3520, 3):
                print('{} shape is wrong: {}'.format(filename, img.shape))
            else:
                imsave(os.path.join(target_dir, '{}_'.format(
                    series)+filename[:-4]+'_ORIG.tif'), img)
                mask = tif_cut(mask, left, right)
                imsave(os.path.join(target_dir, '{}_'.format(series)+filename[:-4]+'_PS.tif'), mask)
                print('{}: {} passed!'.format(idx, filename))
        except:
            print('{} failed!'.format(filename))


def img_split(img, img_PS, folder, index=None, keepALL=True, cut_size=320):
    size_v, size_h = img.shape[0], img.shape[1]
    splits_v = size_v//cut_size
    splits_h = size_h//cut_size

    if keepALL:
        for i in range(0, splits_v, 1):
            for j in range(0, splits_h, 1):
                simg = img[i*cut_size:i*cut_size+cut_size, j*cut_size:j*cut_size+cut_size, :]
                simg_PS = img_PS[i*cut_size:i*cut_size+cut_size, j*cut_size:j*cut_size+cut_size, :]
                if index is None:
                    imsave(folder+'{}{}_ORIG.tif'.format(i, j), simg)
                    imsave(folder+'{}{}_PS.tif'.format(i, j), simg_PS)
                else:
                    imsave(folder+'{}_ORIG.tif'.format(index), simg)
                    imsave(folder+'{}_PS.tif'.format(index), simg_PS)
                    index += 1
    else:
        for i in range(0, splits_v, 1):
            for j in range(0, splits_h, 1):
                simg_PS = img_PS[i*cut_size:i*cut_size+cut_size, j*cut_size:j*cut_size+cut_size, :]
                if simg_PS.min() < 255:
                    simg = img[i*cut_size:i*cut_size+cut_size, j*cut_size:j*cut_size+cut_size, :]
                    if index is None:
                        imsave(folder+'{}{}_ORIG.tif'.format(i, j), simg)
                        imsave(folder+'{}{}_PS.tif'.format(i, j), simg_PS)
                    else:
                        imsave(folder+'{}_ORIG.tif'.format(index), simg)
                        imsave(folder+'{}_PS.tif'.format(index), simg_PS)
                        index += 1
    return index


threshold = 35

img_dir = './old/'
mask_dir = './PS/'
target_dir = './cut/'

preprocess(series, img_dir, mask_dir, target_dir, threshold)

data_dir = './data/'

img_files = [file for file in os.listdir(target_dir) if file.endswith('ORIG.tif')]
mask_files = [file for file in os.listdir(target_dir) if file.endswith('PS.tif')]
print('The number of tifs:', len(img_files)+len(mask_files))
assert len(img_files) == len(mask_files), 'The number of images is not equal to the number of masks!'

index = 0
for img_file in img_files:
    img = plt.imread(os.path.join(target_dir, img_file))
    img_PS = plt.imread(os.path.join(target_dir, img_file.replace('ORIG', 'PS')))
    print(img_file, img.shape, img_PS.shape)
    index = img_split(img, img_PS, data_dir, index, keepALL=False)
    print('Image {} splitting finished!'.format(img_file))
print(index)
