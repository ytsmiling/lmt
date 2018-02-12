import numpy as np
import chainer
from src.function.spectral_norm import SpectralNormFunction
from src.function.spectral_norm_exact import spectral_norm_exact


class Linear(chainer.links.Linear):
    """Overloaded fully-connected layer.
    This also calculates spectral norm when LMT-mode.

    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__(in_size, out_size=out_size, nobias=nobias,
                                     initialW=initialW, initial_bias=initial_bias)
        self.lipschitz = None
        self.u = np.random.random((1, self.out_size)).astype(np.float32) - .5
        self.register_persistent('u')

    def __call__(self, x):
        x, t, l = x
        x = super(Linear, self).__call__(x)
        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                if self.lipschitz is None:
                    self.lipschitz = spectral_norm_exact(self.W.data)
                l *= self.lipschitz
            else:
                self.lipschitz = None
                l *= SpectralNormFunction(self.u)(self.W)
        return x, t, l
