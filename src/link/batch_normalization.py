import numpy as np
import chainer
from src.function import batch_normalization


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

    def __call__(self, x):
        x, t, l = x
        # ensure gamma >= 0
        gamma = chainer.functions.absolute(self.gamma)
        beta = self.beta

        if chainer.config.train:
            func = batch_normalization.BatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, self.decay)
            x, lipschitz = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            mean = chainer.variable.Variable(self.avg_mean)
            var = chainer.variable.Variable(self.avg_var)
            x, lipschitz = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return x, t, l * lipschitz
