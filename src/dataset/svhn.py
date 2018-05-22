import subprocess
import os
import numpy as np
import scipy.io
from chainer.datasets import TupleDataset


def _get_svhn():
    dir_name = os.path.join('dataset', 'svhn')
    os.makedirs(dir_name, exist_ok=True)
    if not os.path.isfile(os.path.join(dir_name, 'train.mat')):
        subprocess.run(['wget',
                        '-O', os.path.join(dir_name, 'train.mat'),
                        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'])
        subprocess.run(['wget',
                        '-O',
                        os.path.join(dir_name, 'test.mat'),
                        'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'])
        subprocess.run(['wget',
                        '-O',
                        os.path.join(dir_name, 'extra.mat'),
                        'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'])


def svhn_small():
    _get_svhn()
    dir_name = os.path.join('dataset', 'svhn')
    train = scipy.io.loadmat(os.path.join(dir_name, 'train.mat'))
    train = TupleDataset(
        train['X'].transpose(3, 2, 0, 1).astype(np.float32) / 255,
        train['y'].flatten().astype(np.int32) - 1)
    test = scipy.io.loadmat(os.path.join(dir_name, 'test.mat'))
    test = TupleDataset(
        test['X'].transpose(3, 2, 0, 1).astype(np.float32) / 255,
        test['y'].flatten().astype(np.int32) - 1)
    return train, test


def svhn():
    """When you start a new project based on this,
    I recommend to modify this function:
    1. randomly select validation set from extra data,
    2. replace test set with the validation set.
    """

    _get_svhn()
    dir_name = os.path.join('dataset', 'svhn')
    train = scipy.io.loadmat(os.path.join(dir_name, 'train.mat'))
    train2 = scipy.io.loadmat(os.path.join(dir_name, 'extra.mat'))
    train_img = np.concatenate(
        [train['X'].transpose(3, 2, 0, 1).astype(np.float32),
         train2['X'].transpose(3, 2, 0, 1).astype(np.float32)]) / 255
    train_label = np.concatenate(
        [train['y'].flatten().astype(np.int32) - 1,
         train2['y'].flatten().astype(np.int32) - 1])
    train = TupleDataset(train_img, train_label)
    test = scipy.io.loadmat(os.path.join(dir_name, 'test.mat'))
    test = TupleDataset(
        test['X'].transpose(3, 2, 0, 1).astype(np.float32) / 255,
        test['y'].flatten().astype(np.int32) - 1)
    return train, test
