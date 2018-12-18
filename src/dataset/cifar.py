import chainer
from chainer.datasets import TupleDataset

means = (0.4914, 0.4822, 0.4465)
sds = (0.2023, 0.1994, 0.2010)

def normalize(X):
    for i in range(3):
        X[:,i,:,:] -= means[i]
        X[:,i,:,:] /= sds[i]
    return X

def transform(dataset):
    X = dataset._datasets[0]
    y = dataset._datasets[1]
    return TupleDataset(normalize(X), y)

def cifar():
    cifar10 = chainer.datasets.get_cifar10()
    train = transform(cifar10[0])
    test = transform(cifar10[1])
    return (train, test)
