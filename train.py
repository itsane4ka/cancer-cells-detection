import os
import argparse
import time

import numpy as np
import math

from models import *
from celldataset import CellImages
from utils import split_train_valid_test, train, test, save_epoch_results
from augmentation import *

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.backends import cudnn


def main():
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--m', default=0.9, type=float, help='Momentum')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--bs', default=16, type=int, help='Batch size')
    parser.add_argument('--cpus', default=8, type=int, help='The number of CPUs for loading data')
    parser.add_argument('--epochs', default=200, type=int, help='The number of training epochs')
    parser.add_argument('--savepred', action='store_true', help='Save predicted masks')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device is {}!'.format(device))
    best_acc = 0.  # best test accuracy
    best_IOU = 0.  # best test IOU
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    image_size = 320

    # Hyperparameters
    batch_size = args.bs
    lr = args.lr
    momentum = args.m
    weight_decay = args.wd
    opt_method = 'SGD_momentum'

    hps = {'opt_method': opt_method, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    print('Batch size: {}, '.format(batch_size) +', '.join([hp+': '+str(value) for hp, value in hps.items()]))

    # Model
    print('==> Building model..')
    net = DeConvNet()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Enabling cudnn, which may lead to about 2 GB extra memory
    if device == 'cuda':
        cudnn.benchmark = True
        print('cudnn benchmark enabled!')

    if args.resume:
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/training_saved.t7')  # Load your saved model
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optim'])
        best_acc = checkpoint['acc']
        best_IOU = checkpoint['metric']
        start_epoch = checkpoint['epoch']
        print('Start Epoch is {}'.format(start_epoch))

    # Data
    print('==> Preparing data..')
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.6672,  0.5865,  0.5985), (1.0, 1.0, 1.0)),
    ])

    joint_transform = JointCompose([
        ArrayToPIL(),
        RandomRotateFlip(),
    ])

    data_dir = './data/'  # specify the data directory if you like

    if os.path.exists('./split_indices/all_indices.txt'):
        all_indices = np.loadtxt('./split_indices/all_indices.txt', dtype=str)
        size = len(all_indices)
        train_indices = all_indices[:-2*int(size/10)]
        valid_indices = all_indices[-2*int(size/10):-int(size/10)]
        test_indices = all_indices[-int(size/10):]

    else:
        train_indices, valid_indices, test_indices = split_train_valid_test(data_dir)
        if not os.path.isdir('split_indices'):
            os.mkdir('split_indices')
        all_indices = train_indices+valid_indices+test_indices
        np.savetxt('./split_indices/all_indices.txt', [int(idx) for idx in all_indices], fmt='%d')

    trainset = CellImages(data_dir, train_indices, img_transform=img_transform, joint_transform=joint_transform)
    print('Trainset size: {}. Number of mini-batch: {}'.format(len(trainset), math.ceil(len(trainset)/batch_size)))

    validset = CellImages(data_dir, valid_indices, img_transform=img_transform)
    print('Validset size: {}. Number of mini-batch: {}'.format(len(validset), math.ceil(len(validset)/batch_size)))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.cpus)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=args.cpus)

    # Training
    print('==> Training begins..')
    for epoch in range(start_epoch, start_epoch+args.epochs):
        start_time = time.time()

        train_results = train(epoch, device, trainloader, net, criterion, optimizer, image_size, is_print_mb=False)
        valid_results = test(epoch, device, validloader, net, criterion, optimizer, image_size, best_acc, best_IOU, hps, is_save=True, is_print_mb=False, is_savepred=args.savepred)
        best_acc, best_IOU = valid_results[-2], valid_results[-1]
        save_epoch_results(epoch, train_results, valid_results, hps)

        print("--- %s seconds ---" % (time.time() - start_time))


# For Windows the conditional statement below is required
if __name__ == '__main__':
    main()
