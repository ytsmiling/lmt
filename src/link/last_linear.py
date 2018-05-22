import numpy as np
import chainer
from src.function.last_fc import LmtFc
from src.function.last_fc import Lmt
from src.function.spectral_norm_exact import spectral_norm_exact
from src.function.l2_norm import l2_norm
from src.hook.power_iteration import register_power_iter
from chainer.functions import linear


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
        u = np.random.normal((1, self.out_size)).astype(np.float32) - .5
        with self.init_scope():
            self.u = chainer.Parameter(u)
            register_power_iter(self.u)
        self.last_fc = True
        self.W.last_fc = True
        self.b.last_fc = True
        self.W.linearW = True

    def __call__(self, x):
        x, t, l = x
        x = super(LastLinear, self).__call__(x)
        if chainer.config.train:
            self.lipschitz = None
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
                    l = l * l2_norm(linear(self.u, self.W))
                if chainer.config.train:
                    x, l = Lmt(t)(x, l)
        return x, t, l
