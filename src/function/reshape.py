from chainer.functions import reshape


def flatten(x):
    x, t, l = x
    x = reshape(x, (x.shape[0], -1))
    return x, t, l
