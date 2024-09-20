import os
import argparse
import time

import numpy as np
import math

from models import *
from celldataset import CellImages
from utils import test

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.backends import cudnn


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--cpus', default=8, type=int, help='The number of CPUs for loading data')
    parser.add_argument('--savepred', action='store_true', help='Save predicted masks')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is {}!'.format(device))
    image_size = 320

    # Hyperparameters
    batch_size = args.bs
    hps = None

    # Model
    print('==> Building model..')
    net = DeConvNet()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = None

    # Enabling cudnn, which may lead to about 2 GB extra memory
    if device == 'cuda':
        cudnn.benchmark = True
        print('cudnn benchmark enabled!')

    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/training_saved.t7')  # Load your saved model
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    best_IOU = checkpoint['metric']

    # Data
    print('==> Preparing data..')
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
    ])

    data_dir = './data/'  # specify the data directory if you like

    all_indices = np.loadtxt('./split_indices/all_indices.txt', dtype=str)
    size = len(all_indices)
    test_indices = all_indices[-int(size/10):]

    testset = CellImages(data_dir, test_indices, img_transform=img_transform)
    print('Testset size: {}. Number of mini-batch: {}'.format(len(testset), math.ceil(len(testset)/batch_size)))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=args.cpus)

    # Evaluation
    print('==> Evaluation begins..')
    start_time = time.time()

    test_results = test(epoch, device, testloader, net, criterion, optimizer, image_size, best_acc, best_IOU, hps, is_save=False, is_print_mb=False, is_savepred=args.savepred)

    print("--- %s seconds ---" % (time.time() - start_time))


# For Windows the conditional statement below is required
if __name__ == '__main__':
    main()
