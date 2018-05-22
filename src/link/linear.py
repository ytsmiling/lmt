import numpy as np
import chainer
import chainer.functions as F
from src.function.spectral_norm_exact import spectral_norm_exact
from src.function.normalize import normalize
from src.function.l2_norm import l2_norm
from src.hook.power_iteration import register_power_iter


class Linear(chainer.links.Linear):
    """Overloaded fully-connected layer.
    This also calculates spectral norm when LMT-mode.

    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__(in_size, out_size=out_size, nobias=nobias,
                                     initialW=initialW,
                                     initial_bias=initial_bias)
        self.lipschitz = None
        self.W.linearW = True
        self.u = None

    def __call__(self, x):
        x, t, l = x
        if chainer.config.train:
            self.lipschitz = None
        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                if self.lipschitz is None:
                    self.lipschitz = spectral_norm_exact(self.W.data)
                l = l * self.lipschitz
                x = super(Linear, self).__call__(x)
            else:
                if self.u is None:
                    # for calculation of Lipschitz constant
                    u = np.random.normal(
                        size=(1, x.shape[1])).astype(np.float32)
                    with self.init_scope():
                        self.u = chainer.Parameter(u)
                        register_power_iter(self.u)
                    if self._device_id is not None and self._device_id >= 0:
                        with chainer.cuda._get_device(self._device_id):
                            self.u.to_gpu()
                x = super(Linear, self).__call__(x)
                normalize(self.u.array)
                u = F.linear(self.u, self.W)
                l = l * l2_norm(u)
        else:
            x = super(Linear, self).__call__(x)
        return x, t, l
