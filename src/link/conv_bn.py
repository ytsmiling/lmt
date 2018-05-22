import math
import numpy as np
import chainer
import chainer.functions as F
from src.function.normalize import normalize
from src.function.conv_spectral_norm_exact import conv_spectral_norm_exact
from src.function.l2_norm import l2_norm
from src.hook.power_iteration import register_power_iter


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
                                              initialW=initialW,
                                              initial_bias=initial_bias,
                                              **kwargs)
        self.lipschitz = None
        self.parseval_factor = None
        self.u = np.random.random((1, self.out_channels)).astype(
            np.float32) - .5
        self.register_persistent('u')

        self.avg_mean = np.zeros(out_channels, dtype=np.float32)
        self.register_persistent('avg_mean')
        self.avg_var = np.zeros(out_channels, dtype=np.float32)
        self.register_persistent('avg_var')
        self.decay = decay
        self.eps = eps
        self.W.convolutionW = True
        self.u = None

        with self.init_scope():
            self.gamma = chainer.Parameter(np.ones((out_channels,), np.float32))
            self.beta = chainer.Parameter(np.zeros((out_channels,), np.float32))

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
        gamma = self.gamma
        beta = self.beta

        x = super(Convolution2DBN, self).__call__(x_in)
        if getattr(chainer.config, 'parseval', False):
            # in Parseval networks, output is rescaled
            x = x * self.parseval_factor
            l = l * self.parseval_factor

        reshape = (1, x.shape[1], 1, 1)

        if chainer.config.train:
            # batch norm
            mean = F.mean(x, axis=(0,) + tuple(range(2, x.ndim)))
            x = x - F.broadcast_to(
                F.reshape(mean, reshape),
                x.shape)
            var = F.mean(x ** 2, axis=(0,) + tuple(range(2, x.ndim)))
            m = x.size // self.gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean.array
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var.array
        else:
            mean = self.avg_mean
            var = self.avg_var
            x = x - F.broadcast_to(F.reshape(mean, reshape), x.shape)

        z0 = F.identity(self.gamma) / F.sqrt(var + self.eps)
        z = F.reshape(z0, reshape)
        x = x * F.broadcast_to(z, x.shape) + F.broadcast_to(
            F.reshape(self.beta, reshape), x.shape)

        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                # inference with calculation of Lipschitz constant
                # (all configuration)
                if self.lipschitz is None:
                    W = self.W.array * (gamma.array
                                        / self.xp.sqrt(var + self.eps)
                                        )[:, None, None, None]
                    self.lipschitz = conv_spectral_norm_exact(
                        W, self.u.shape, self.stride, self.pad)
                l = l * self.lipschitz
                return x, t, l

        if getattr(chainer.config, 'lmt', False):
            # lmt training and non-exact inference
            # lipschitz
            normalize(self.u.array)
            # this is practically faster than concatenation
            u = super(Convolution2DBN, self).__call__(self.u)
            u = u * F.broadcast_to(z, u.shape)
            l = l * l2_norm(u)

            return x, t, l

        # training and inference for other settings
        # we do not have to calculate l (since it will not be used)
        return x, t, l
