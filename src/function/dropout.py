import math
import numpy
import chainer
from chainer import cuda
from chainer import function


def dropout(x, ratio=.5, **kwargs):
    """dropout regularization
    Even though it scales its input at training,
    we do not consider it in Lipschitz constant.

    :param x: input (vector/tensor, label, lipschitz)
    :param ratio: dropout ratio
    :return:
    """
    x, t, l = x
    x = chainer.functions.dropout(x, ratio=ratio, **kwargs)
    return x, t, l


class PartialDrop(function.Function):
    def __init__(self, ratio=.5, partial=.5):
        self.ratio = ratio
        self.partial = partial
        self.index = 0

    def forward(self, x):
        self.retain_inputs(())
        self.index = int(math.trunc(x[0].shape[1] * self.ratio))
        if not hasattr(self, 'mask'):
            scale = x[0].dtype.type(1. / (1 - self.ratio))
            xp = cuda.get_array_module(*x)
            flag = (xp.random.rand(*x[0].shape, dtype=numpy.float32) >=
                    self.ratio)
            self.mask = scale * flag
            self.mask[:, :self.index] = 1.
        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def partial_drop(x, ratio=.5, partial=.5):
    x, t, l = x
    x = PartialDrop(ratio=ratio, partial=partial)(x)
    return x, t, l
