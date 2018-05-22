import math
import numpy as np
import chainer
import chainer.functions as F
from src.function.conv_spectral_norm_exact import conv_spectral_norm_exact
from src.function.normalize import normalize
from src.function.l2_norm import l2_norm
from src.hook.power_iteration import register_power_iter


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
                                            initialW=initialW,
                                            initial_bias=initial_bias,
                                            **kwargs)
        self.W.convolutionW = True
        self.lipschitz = None
        self.parseval_factor = None
        self.u = None

    def __call__(self, x):
        x_in, t, l = x
        if chainer.config.train:
            self.lipschitz = None
        if self.parseval_factor is None:
            k_h, k_w = (self.ksize if isinstance(self.ksize, tuple)
                        else (self.ksize, self.ksize))
            # rescaling factor of Parseval networks
            # According to the author, this factor is not essential
            self.parseval_factor = 1 / math.sqrt(k_h * k_w)
        if self.u is None:
            # for calculation of Lipschitz constant
            u = np.random.normal(size=(1,) + x_in.shape[1:]).astype(np.float32)
            with self.init_scope():
                self.u = chainer.Parameter(u)
                register_power_iter(self.u)
            if self._device_id is not None and self._device_id >= 0:
                with chainer.cuda._get_device(self._device_id):
                    self.u.to_gpu()

        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                # inference with calculation of Lipschitz constant
                # (all configuration)
                x = super(Convolution2D, self).__call__(x_in)
                if self.lipschitz is None:
                    self.lipschitz = conv_spectral_norm_exact(
                        self.W.array, self.u.shape, self.stride, self.pad)
                if getattr(chainer.config, 'parseval', False):
                    # in Parseval networks, output is rescaled
                    x = x * self.parseval_factor
                    l = l * self.parseval_factor
                l = l * self.lipschitz
                return x, t, l

        if getattr(chainer.config, 'lmt', False):
            # lmt training and non-exact inference
            normalize(self.u.array)
            # this is practically faster than concatenation
            x = super(Convolution2D, self).__call__(x_in)
            u = F.convolution_2d(self.u, self.W, stride=self.stride,
                                 pad=self.pad)
            l = l * l2_norm(u)
            return x, t, l

        # training and inference for other settings
        x = super(Convolution2D, self).__call__(x_in)
        if getattr(chainer.config, 'parseval', False):
            # in Parseval networks, output is rescaled
            x = x * self.parseval_factor

        # we do not have to calculate l (since it will not be used)
        return x, t, l
