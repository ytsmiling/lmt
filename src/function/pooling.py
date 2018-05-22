import math
import chainer


def to_tuple(cand):
    if isinstance(cand, tuple):
        return cand
    else:
        return cand, cand


def factor(shape, ksize, stride, pad):
    ksize = to_tuple(ksize)
    stride = to_tuple(stride)
    pad = to_tuple(pad)
    l = math.ceil(min(ksize[0], shape[2] + pad[0] * 2 - ksize[0] + 1) / stride[0])
    l *= math.ceil(min(ksize[1], shape[3] + pad[1] * 2 - ksize[1] + 1) / stride[1])
    return math.sqrt(l)


def max_pooling_2d(x, ksize, stride, pad=0, cover_all=True):
    x, t, l = x
    l *= factor(x.shape, ksize, stride, pad)
    x = chainer.functions.max_pooling_2d(
        x, ksize=ksize, stride=stride, pad=pad, cover_all=cover_all)
    return x, t, l


def average_pooling_2d(x, ksize, stride, pad=0):
    x, t, l = x
    l *= factor(x.shape, ksize, stride, pad)
    ksize = to_tuple(ksize)
    l /= math.sqrt(ksize[0] * ksize[1])
    x = chainer.functions.average_pooling_2d(
        x, ksize=ksize, stride=stride, pad=pad)
    return x, t, l
