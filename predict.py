import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

from models import *
from utils import predict, img_split

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('--bs', default=10, type=int, help='Batch size = image width/320')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is {}!'.format(device))

    # Hyperparameters
    batch_size = args.bs
    image_size = 320
    print('Batch size: {}'.format(batch_size))

    # Model
    print('==> Building model..')
    net = DeConvNet()
    net = net.to(device)

    # Enabling cudnn, which may lead to about 2 GB extra memory
    if device == 'cuda':
        cudnn.benchmark = True
        print('cudnn benchmark enabled!')

    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/training_saved.t7')  # Load your saved model
    net.load_state_dict(checkpoint['net'])

    # Tranformation
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
    ])

    X_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.transforms.Pad((160, 0, 160, 0), fill=(0, 0, 0), padding_mode='constant'),
    ])

    Y_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.transforms.Pad((0, 160, 0, 160), fill=(0, 0, 0), padding_mode='constant'),
    ])

    XY_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.transforms.Pad((160, 160, 160, 160), fill=(0, 0, 0), padding_mode='constant'),
    ])

    # Prediction
    print('==> Prediction begins..')
    net.eval()
    with torch.no_grad():
        for photo_folder in os.listdir('./photo/'):
            img_files = [os.path.join(photo_folder, file) for file in os.listdir(
                os.path.join('./photo/', photo_folder)) if file.endswith("ORIG.tif")]
            for img_file in img_files:
                start_time = time.time()
                
                print('{}:'.format(img_file))
                img = plt.imread(os.path.join('./photo/', img_file))

                simgs = img_split(img, cut_size=image_size)
                simgs_X = img_split(np.asarray(X_transform(img)), cut_size=image_size)
                simgs_Y = img_split(np.asarray(Y_transform(img)), cut_size=image_size)
                simgs_XY = img_split(np.asarray(XY_transform(img)), cut_size=image_size)

                mask = predict(device, net, img_transform, simgs, overlap_mode=0, batch_size=batch_size, image_size=image_size).astype(bool)
                mask_X = predict(device, net, img_transform, simgs_X, overlap_mode=1, batch_size=batch_size+1, image_size=image_size).astype(bool)
                mask_Y = predict(device, net, img_transform, simgs_Y, overlap_mode=2, batch_size=batch_size, image_size=image_size).astype(bool)
                mask_XY = predict(device, net, img_transform, simgs_XY, overlap_mode=3, batch_size=batch_size+1, image_size=image_size).astype(bool)

                mask_combined = (mask+mask_X+mask_Y+mask_XY).astype(np.uint8)
                imsave('./photo/{}_PRED.tif'.format(img_file[:-9]), (1-mask_combined)*255)
                print('The mask of {} was predicted and saved!'.format(img_file[:-9]))
                
                print("--- %s seconds ---" % (time.time() - start_time))
            print('{} complete!\n'.format(photo_folder))


# For Windows the conditional statement below is required
if __name__ == '__main__':
    main()
