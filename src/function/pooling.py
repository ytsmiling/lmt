import math
import chainer


def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True):
    x, t, l = x
    assert x.shape[2] == x.shape[3]
    assert isinstance(ksize, int)
    assert isinstance(stride, int)
    assert isinstance(pad, int)
    l *= math.ceil(min(ksize, x.shape[2] + pad * 2 - ksize + 1) / stride)
    x = chainer.functions.max_pooling_2d(
        x, ksize=ksize, stride=stride, pad=pad, cover_all=cover_all)
    return x, t, l


def average_pooling_2d(x, ksize, stride=None, pad=0):
    x, t, l = x
    assert x.shape[2] == x.shape[3]
    assert isinstance(ksize, int)
    assert isinstance(stride, int)
    assert isinstance(pad, int)
    l *= math.ceil(min(ksize, x.shape[2] + pad * 2 - ksize + 1) / stride) / ksize
    x = chainer.functions.average_pooling_2d(
        x, ksize=ksize, stride=stride, pad=pad)
    return x, t, l
