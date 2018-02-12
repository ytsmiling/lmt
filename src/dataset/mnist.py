import chainer


def mnist():
    train, test = chainer.datasets.get_mnist()
    train._datasets = train._datasets[0] * 255, train._datasets[1]
    test._datasets = test._datasets[0] * 255, test._datasets[1]
    return train, test
