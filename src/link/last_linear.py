import numpy as np
import chainer
from src.function.last_fc import LmtFc
from src.function.last_fc import Lmt
from src.function.spectral_norm import SpectralNormFunction
from src.function.spectral_norm_exact import spectral_norm_exact


class LastLinear(chainer.links.Linear):
    """Overloaded fully-connected layer.
    This Link is expected to be used in the last fully-connected layer.
    This inculdes special procedure for lmt and lmt++.
    They are triggered depending on values in chainer.config.

    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(LastLinear, self).__init__(in_size, out_size=out_size, nobias=nobias,
                                         initialW=initialW, initial_bias=initial_bias)
        self.lipschitz = None
        self.u = np.random.random((1, self.out_size)).astype(np.float32) - .5
        self.register_persistent('u')
        self.last_fc = True
        self.W.last_fc = True
        self.b.last_fc = True

    def __call__(self, x):
        x, t, l = x
        x = super(LastLinear, self).__call__(x)
        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'lipschitz_regularization', False):
                _, l = LmtFc(t)(x, self.W, l)
            elif getattr(chainer.config, 'lmt-fc', False):
                if chainer.config.train:
                    x, l = LmtFc(t)(x, self.W, l)
                else:
                    _, l = LmtFc(t)(x, self.W, l)
            else:
                if getattr(chainer.config, 'exact', False):
                    if self.lipschitz is None:
                        self.lipschitz = spectral_norm_exact(self.W)
                    l = l * self.lipschitz
                else:
                    self.lipschitz = None
                    l = l * SpectralNormFunction(self.u)(self.W)
                if chainer.config.train:
                    x, l = Lmt(t)(x, l)
        return x, t, l
