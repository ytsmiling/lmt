import math
import chainer
from chainer import cuda
from chainer import function


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


class PartialReLU(function.Function):
    def __init__(self, ratio=.5):
        self.ratio = ratio
        self.index = 0

    def forward_gpu(self, x):
        self.retain_inputs(())
        self.index = int(math.trunc(x[0].shape[1] * self.ratio))
        y = cuda.cupy.maximum(x[0], 0)
        y[:, self.index:] = x[0][:, self.index:]
        self.retain_outputs((0,))
        return y,

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = y > 0 ? gy : (T)0',
            'relu_bwd')(y, gy[0])
        gx[:, self.index:] = gy[0][:, self.index:]
        return gx,


def partial_relu(x, ratio=.5):
    x, t, l = x
    x = PartialReLU(ratio=ratio)(x)
    return x, t, l
