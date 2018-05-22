import numpy as np
import chainer
import chainer.functions as F
from src.function.normalize import normalize
from src.function.perturb import perturb
from src.function.l2_norm import l2_norm
from src.hook.power_iteration import register_power_iter


class BatchNormalization(chainer.link.Link):
    """Batch-normalization layer.

    """

    def __init__(self, size, decay=0.9, eps=2e-5, *args, **kwargs):
        super(BatchNormalization, self).__init__()
        self.avg_mean = np.zeros(size, dtype=np.float32)
        self.register_persistent('avg_mean')
        self.avg_var = np.zeros(size, dtype=np.float32)
        self.register_persistent('avg_var')
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            self.gamma = chainer.Parameter(np.ones((size,), np.float32))
            self.beta = chainer.Parameter(np.zeros((size,), np.float32))
            u = np.random.normal(size=size).astype(np.float32)
            with self.init_scope():
                self.u = chainer.Parameter(u)
                register_power_iter(self.u)

    def __call__(self, x):
        x, t, l = x

        reshape = (1, x.shape[1]) + (1,) * (x.ndim - 2)

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

        # calculate Lipschitz constant
        if getattr(chainer.config, 'lmt', False):
            if getattr(chainer.config, 'exact', False):
                l = l * F.reshape(F.max(F.absolute(z0)), (1,))
            else:
                normalize(self.u.array)
                perturb(self.u.array, 1e-2, self.xp)
                u = self.u * z0
                l = l * l2_norm(u)

        return x, t, l
