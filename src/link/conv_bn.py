import math
import numpy as np
import chainer
from src.function import conv_bn
from src.function.spectral_norm_exact import spectral_norm_exact
from src.function.spectral_norm import SpectralNormFunction


class Convolution2DBN(chainer.links.Convolution2D):
    """This is a combined layer conv-2d + batch-norm.
    This also calculates its spectral norm in LMT-mode.

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 initialW=None, initial_bias=None,
                 decay=.9, eps=2e-5, nobias=True, **kwargs):
        super(Convolution2DBN, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            ksize=ksize, stride=stride,
                                            pad=pad, nobias=True,
                                            initialW=initialW, initial_bias=initial_bias,
                                            **kwargs)
        self.lipschitz = None
        self.factor = None
        self.u = np.random.random((1, self.out_channels)).astype(np.float32) - .5
        self.register_persistent('u')

        self.avg_mean = np.zeros(out_channels, dtype=np.float32)
        self.register_persistent('avg_mean')
        self.avg_var = np.zeros(out_channels, dtype=np.float32)
        self.register_persistent('avg_var')
        self.decay = decay
        self.eps = eps
        self.u = np.random.random((1, self.out_channels)).astype(np.float32) - .5
        self.register_persistent('u')

        with self.init_scope():
            self.gamma = chainer.Parameter(np.ones((out_channels,), np.float32))
            self.beta = chainer.Parameter(np.zeros((out_channels,), np.float32))

    def __call__(self, x):
        x_in, t, l = x
        if chainer.config.train:
            self.lipschitz = None

        # convolution
        x = super(Convolution2DBN, self).__call__(x_in)

        if self.factor is None:
            #
            # calculation of \|U\|_2
            #
            k_out, k_in, k_h, k_w = self.W.shape
            s_h, s_w = self.stride
            p_h, p_w = self.pad
            # x_in's shape is (batchsize, k_in, h, w)
            assert x_in.shape[1] == k_in
            self.factor = math.ceil(min(k_h, x_in.shape[2] + p_h * 2 - k_h + 1) / s_h)
            self.factor *= math.ceil(min(k_w, x_in.shape[3] + p_w * 2 - k_w + 1) / s_w)
            self.factor = math.sqrt(self.factor)

            #
            # rescaling factor of Parseval networks
            # According to the author, this factor is not essential
            #
            self.parseval = 1 / math.sqrt(k_h * k_w)

        l = l * self.factor

        # rescaling of Parseval networks
        if getattr(chainer.config, 'parseval', False):
            x = x * self.parseval
            l = l * self.parseval

        # ensure that gamma >= 0
        gamma = chainer.functions.absolute(self.gamma)
        beta = self.beta

        # batch norm
        if chainer.config.train:
            func = conv_bn.BatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, self.decay, self.u)
            x, lipschitz = func(x, gamma, beta, self.W)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            mean = chainer.variable.Variable(self.avg_mean)
            var = chainer.variable.Variable(self.avg_var)
            x, lipschitz = conv_bn.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, self.W, self.u)

        if getattr(chainer.config, 'lmt', False) and getattr(chainer.config, 'exact', False):
            assert not chainer.config.train
            if self.lipschitz is None:
                W = (gamma.data / self.xp.sqrt(var.data + self.eps)
                     ).reshape((self.W.shape[0], 1)) * self.W.data.reshape((self.W.shape[0], -1))
                self.lipschitz = spectral_norm_exact(W)
            l = l * self.lipschitz
        else:
            l = l * lipschitz

        return x, t, l
