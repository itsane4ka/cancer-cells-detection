# Plot the loss, accuracy and IOU of each epoch

import matplotlib.pyplot as plt
import numpy as np

dir = '' # Specify the directory of your saved results for each epoch 
epoch, train_loss, train_acc, train_IOU, test_loss, test_acc, test_IOU = np.loadtxt(
    dir, comments=',', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)

opt_method = 'SGD momentum'
lr = 0.01

print(test_loss.argmin())
print(test_acc.argmax())
print(test_IOU.argmax())

is_train_loss = False
is_test_loss = False

is_train_acc = False
is_test_acc = False

is_train_IOU = True
is_test_IOU = True

plt.figure()
if is_train_loss:
    plt.plot(epoch, train_loss, 'b', label='{} lr={} train loss'.format(opt_method, lr))

if is_test_loss:
    plt.plot(epoch, test_loss, 'r', label='{} lr={} test loss'.format(opt_method, lr))

if is_train_acc:
    plt.plot(epoch, train_acc, 'b', label='{} lr={} train acc'.format(opt_method, lr))

if is_test_acc:
    plt.plot(epoch, test_acc, 'r', label='{} lr={} test acc'.format(opt_method, lr))

if is_train_IOU:
    plt.plot(epoch, train_IOU, 'b', label='{} lr={} train IOU'.format(opt_method, lr))

if is_test_IOU:
    plt.plot(epoch, test_IOU, 'r', label='{} lr={} test IOU'.format(opt_method, lr))

plt.xlabel('Epoch')

if is_train_loss or is_test_loss:
    plt.ylabel('Cross Entropy Loss')
    # plt.ylim(0., 1.)
    plt.legend(loc='upper right')

if is_train_acc or is_test_acc:
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

if is_train_IOU or is_test_IOU:
    plt.ylabel('IOU')
    plt.legend(loc='lower right')

plt.show()
