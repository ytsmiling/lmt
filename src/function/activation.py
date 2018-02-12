import chainer


def relu(x):
    x, t, l = x
    x = chainer.functions.relu(x)
    return x, t, l


def sigmoid(x):
    x, t, l = x
    x = chainer.functions.sigmoid(x)
    l = l * 0.25
    return x, t, l


def tanh(x):
    x, t, l = x
    x = chainer.functions.tanh(x)
    return x, t, l
