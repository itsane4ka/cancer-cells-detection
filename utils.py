import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

import torch
from torch.utils.data import DataLoader


def split_train_valid_test(data_dir):
    tifs = [file for file in os.listdir(data_dir) if (
        file.endswith("ORIG.tif") or file.endswith("PS.tif"))]
    size = len(tifs)//2
    valid_size = int(size/10)
    test_size = int(size/10)
    
    test_indices = np.random.choice(range(size), test_size, replace=False)
    rest_indices = np.array(list(set(range(size))-set(test_indices)))

    valid_indices = np.random.choice(rest_indices, valid_size, replace=False)
    train_indices = np.array(list(set(rest_indices)-set(valid_indices)))
    np.random.shuffle(train_indices)

    train_indices, valid_indices, test_indices = list(map(str, train_indices)), list(
        map(str, valid_indices)), list(map(str, test_indices))
    return train_indices, valid_indices, test_indices


def split_cross_validation(data_dir, splits=8):
    tifs = [file for file in os.listdir(data_dir) if (
        file.endswith("ORIG.tif") or file.endswith("PS.tif"))]
    size = len(tifs)//2
    all_indices = np.arange(size)
    np.random.shuffle(all_indices)

    test_size = int(size/splits)
    cv_indices = []
    for i in range(splits):
        test_indices = all_indices[i*test_size:(i+1)*test_size]
        train_indices = np.delete(all_indices, np.s_[i*test_size:(i+1)*test_size])
        train_indices, test_indices = list(map(str, train_indices)), list(map(str, test_indices))
        cv_indices.append((train_indices, test_indices))
    return cv_indices


def get_mean(dataset):
    '''Compute the mean value of dataset.'''
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    print('==> Computing mean..')
    for idx, sample in enumerate(dataloader):
        if idx % 100 == 0:
            print('{} samples finished calculation.'.format(idx))
        input = sample[0]
        for i in range(3):
            mean[i] += input[:, i, :, :].mean()
    mean.div_(len(dataset))
    return mean


def show_img_and_mask(img, mask):
    img[0].add_(0.6672)
    img[1].add_(0.5865)
    img[2].add_(0.5985)
    npimg, npmask = img.numpy(), mask.numpy()
    img = (np.transpose(npimg, (1, 2, 0))*255).astype(np.uint8)
    mask = np.transpose(npmask, (1, 2, 0))
    both = np.concatenate((img, mask), axis=1)
    plt.imshow(both)
    plt.show()


def progress_bar(msg):
    pass


def reinitialize_net(net):
    count = 0
    for child in net.children():
        if hasattr(child, 'reset_parameters'):
            child.reset_parameters()
            count += 1
        elif isinstance(child, torch.nn.Sequential):
            for ly in child:
                if hasattr(ly, 'reset_parameters'):
                    ly.reset_parameters()
                    count += 1
    print('{} layers have been initialized!'.format(count))


def save_epoch_stats(epoch, train_results, test_results, hps):
    if not os.path.isdir('epoch_results'):
        os.mkdir('epoch_results')

    name = '-'.join([hp+str(value) for hp, value in hps.items()])+'.txt'
    with open('./epoch_results/'+name, 'a') as resfile:
        resfile.write('{0:} {1:.4f} {2:.2f} {3:.2f} {4:.4f} {5:.2f} {6:.2f}'.format(
            epoch, train_results[0], train_results[1], train_results[2], test_results[0], test_results[1], test_results[2]))
        resfile.write('\n')


def train(epoch, device, trainloader, net, criterion, optimizer, image_size, is_print_mb=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    IOUs = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item()*targets.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted.eq(targets).sum().item()/(image_size**2))

        intersect = predicted*targets
        union = predicted+targets-intersect
        intersect = intersect.sum(2).sum(1).cpu().numpy()
        union = union.sum(2).sum(1).cpu().numpy()
        zeros = union == 0
        intersect[zeros] = 1
        union[zeros] = 1
        IOUs += (intersect/union).sum()

        if is_print_mb and batch_idx % 100 == 0:
            print('minibatch: {0:3};  cur_Loss: {1:.4f};  cur_Acc: {2:.2f};  IOU: {3:.2f}'.format(
                batch_idx, train_loss/total, 100.*correct/total, 100.*IOUs/total))
                
    print('Epoch (training) complete. Loss: {0:.4f};  Acc: {1:.2f};  IOU: {2:.2f}'.format(
        train_loss/total, 100.*correct/total, 100.*IOUs/total))
    return train_loss/total, 100.*correct/total, 100.*IOUs/total


def test(epoch, device, testloader, net, criterion, optimizer, image_size, best_acc, best_IOU, hps, is_savenet=True, is_print_mb=True, is_savepred=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    IOUs = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += (loss.item()*targets.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted.eq(targets).sum().item()/(image_size**2))

            intersect = predicted*targets
            union = predicted+targets-intersect
            intersect = intersect.sum(2).sum(1).cpu().numpy()
            union = union.sum(2).sum(1).cpu().numpy()
            zeros = union == 0
            intersect[zeros] = 1
            union[zeros] = 1
            IOUs += (intersect/union).sum()

            if is_savepred:
                if not os.path.isdir('prediction'):
                    os.mkdir('prediction')
                prediction = predicted.cpu().numpy()
                for i in range(len(prediction)):
                    mask_pred = prediction[i].astype(np.uint8)
                    imsave('./prediction/{}_{}.tif'.format(batch_idx, i), (1-mask_pred)*255)

            if is_print_mb and batch_idx % 100 == 0:
                print('minibatch: {0:3};  cur_Loss: {1:.4f};  cur_Acc: {2:.2f};  IOU: {3:.2f}'.format(
                    batch_idx, test_loss/total, 100.*correct/total, 100.*IOUs/total))

        print('Epoch (test) complete. Loss: {0:.4f};  Acc: {1:.2f};  IOU: {2:.2f}'.format(
            test_loss/total, 100.*correct/total, 100.*IOUs/total))

    # Save checkpoint
    acc = correct/total
    metric = IOUs/total

    if acc > best_acc and is_savenet:
        print('Saving (metric is pixel accuracy)..')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc,
            'metric': metric,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        name = '-'.join([hp+str(value) for hp, value in hps.items()])+'-best_acc.t7'
        torch.save(state, './checkpoint/'+name)
        best_acc = acc

    if metric > best_IOU and is_savenet:
        print('Saving (metric is IoU)..')
        state = {
            'net': net.state_dict(),
            'optim': optimizer.state_dict(),
            'acc': acc,
            'metric': metric,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        name = '-'.join([hp+str(value) for hp, value in hps.items()])+'-best_IOU.t7'
        torch.save(state, './checkpoint/'+name)
        best_IOU = metric
        
    return test_loss/total, 100.*correct/total, 100.*IOUs/total, best_acc, best_IOU


def predict(device, net, img_transform, simgs, overlap_mode=0, batch_size=11, image_size=320):
    mask_pred = []

    for i in range(len(simgs)//batch_size):
        batch_imgs = []
        for img_np in simgs[i*batch_size:i*batch_size+batch_size]:
            simg = img_transform(img_np)
            simg = simg.reshape(1, *simg.shape)
            batch_imgs.append(simg)
        inputs = torch.cat(tuple(batch_imgs))

        inputs = inputs.to(device)
        outputs = net(inputs)

        _, predicted = outputs.max(1)

        prediction = predicted.cpu().numpy()
        batch_masks = prediction.astype(np.uint8)
        mask_pred.append(np.concatenate(tuple(batch_masks), axis=-1))

    mask_pred = np.array(mask_pred).reshape(
        len(simgs)//batch_size*image_size, batch_size*image_size)

    if overlap_mode == 1:
        mask_pred = mask_pred[:, image_size//2:-image_size//2]
    elif overlap_mode == 2:
        mask_pred = mask_pred[image_size//2:-image_size//2, :]
    elif overlap_mode == 3:
        mask_pred = mask_pred[image_size//2:-image_size//2, image_size//2:-image_size//2]

    return mask_pred


def img_split(img, cut_size=320):
    size_v, size_h = img.shape[0], img.shape[1]
    splits_v = size_v//cut_size
    splits_h = size_h//cut_size

    simgs = []
    for i in range(0, splits_v, 1):
        for j in range(0, splits_h, 1):
            simg = img[i*cut_size:i*cut_size+cut_size, j*cut_size:j*cut_size+cut_size, :]
            simgs.append(simg)
    return simgs
