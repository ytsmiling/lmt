import chainer
from chainer.cuda import cupy as cp


class SpectralNormRegularization(object):
    """Spectral norm regularization.
    Note: they did not use weight decay for bias/batch-normalization parameters.

    """

    name = 'SpectralNormRegularization'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        if param.ndim >= 2:
            xp = chainer.cuda.get_array_module(param.data)
            if not hasattr(param, 'snr_vector'):
                param.snr_vector = xp.random.random(
                    (1, param.shape[0])).astype(xp.float32)
            param.snr_vector /= cp.linalg.norm(param.snr_vector)
            W = param.data.reshape((param.shape[0], -1))
            u = param.snr_vector.dot(W)
            v = u.dot(W.T)
            sigma = cp.linalg.norm(u) / cp.linalg.norm(v)
            param.grad += self.rate * sigma * (
                v.reshape((v.size, 1)) * u).reshape(param.shape)
            param.snr_vector[:] = v
