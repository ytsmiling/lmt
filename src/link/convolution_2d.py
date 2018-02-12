import math
import numpy as np
import chainer
from src.function.spectral_norm import SpectralNormFunction
from src.function.spectral_norm_exact import spectral_norm_exact


class Convolution2D(chainer.links.Convolution2D):
    """Overloaded 2-dimensional convolution layer.
    This also calculates spectral norm when LMT-mode.

    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, **kwargs):
        super(Convolution2D, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            ksize=ksize, stride=stride,
                                            pad=pad, nobias=nobias,
                                            initialW=initialW, initial_bias=initial_bias,
                                            **kwargs)
        self.lipschitz = None
        self.factor = None
        self.u = np.random.random((1, self.out_channels)).astype(np.float32) - .5
        self.register_persistent('u')

    def __call__(self, x):
        x_in, t, l = x
        x = super(Convolution2D, self).__call__(x_in)
        if chainer.config.train:
            self.lipschitz = None

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

        if getattr(chainer.config, 'parseval', False):
            x = x * self.parseval
            l = l * self.parseval

        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                if self.lipschitz is None:
                    self.lipschitz = spectral_norm_exact(self.W.data)
                l = l * self.factor * self.lipschitz
            else:
                if not getattr(chainer.config, 'parseval', False):
                    l = l * self.factor * SpectralNormFunction(self.u)(self.W)
        return x, t, l
