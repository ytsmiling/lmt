import numpy as np
import chainer
from chainer.cuda import get_array_module
from chainer.functions import convolution_2d, deconvolution_2d
from chainer.backends import cuda
from src.function.pooling import factor

if cuda.available:
    def normalize(arr, axis):
        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_conv')(arr, axis=axis, keepdims=True)
        cuda.elementwise('T norm',
                         'T x',
                         'x /= (norm + 1e-20)',
                         'div_conv_norm')(norm, arr)
        return norm
else:
    def normalize(arr, axis):
        norm = np.sqrt((arr ** 2).sum(axis, keepdims=True))
        arr /= norm + 1e-20
        return norm


def conv_spectral_norm_exact(kernel, shape, stride, pad):
    xp = get_array_module(kernel)
    kernel = kernel.astype(xp.float64)
    shape = (128,) + shape[1:]
    x = xp.random.normal(size=shape).astype(xp.float64)
    normalize(x, (1, 2, 3))
    prev = None
    eps = 1e20
    with chainer.no_backprop_mode():
        for i in range(5000):
            x = convolution_2d(x, kernel, stride=stride, pad=pad).array
            x = deconvolution_2d(x, kernel, stride=stride, pad=pad).array
            norm = normalize(x, (1, 2, 3))
            if prev is not None:
                eps = norm - prev
            prev = norm
        f = xp.abs(eps) * np.prod(shape[1:])
        error = (f + xp.sqrt(f * (4 * prev + f))) / 2
    return xp.sqrt(xp.max(prev + error))


def conv_spectral_norm_improved(kernel, shape, stride, pad):
    xp = get_array_module(kernel)
    s, v, d = xp.linalg.svd(kernel.reshape(kernel.shape[0], -1))
    return xp.max(v) * factor(shape, kernel.shape[2:], stride, pad)


def conv_spectral_norm_parseval(kernel, shape, stride, pad):
    xp = get_array_module(kernel)
    s, v, d = xp.linalg.svd(kernel.reshape(kernel.shape[0], -1))
    factor = xp.sqrt(kernel.shape[2] * kernel.shape[3])
    return xp.max(v) * factor


def conv_frobenius_norm(kernel, shape, stride, pad):
    return (kernel ** 2).sum()
