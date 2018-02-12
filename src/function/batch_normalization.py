import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import argument


class BatchNormalizationFunction(function.Function):
    """Calculates batch normalization.

    """

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9):
        self.running_mean = mean
        self.running_var = var
        self.eps = eps
        self.decay = decay

    def forward(self, inputs):
        #
        # preprocessing
        #
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if configuration.config.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros_like(beta, dtype=xp.float32)
                self.running_var = xp.zeros_like(gamma, dtype=xp.float32)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = beta.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)

        #
        # start of forward path
        #
        if configuration.config.train:
            axis = (0,) + tuple(range(head_ndim, x.ndim))
            mean = x.mean(axis=axis)
            var = cuda.reduce(
                'S x, T mean, T alpha', 'T out',
                '(x - mean) * (x - mean)',
                'a + b', 'out = alpha * a', '0', 'bn_var')(
                x, mean[expander], x.shape[1] / x.size, axis=axis, keepdims=False
            )
        else:
            mean = self.fixed_mean
            var = self.fixed_var
        if xp is numpy:
            raise NotImplementedError()
        else:
            self.std_inv = cuda.elementwise(
                'T var, T eps', 'T std_inv',
                '''
                std_inv = 1 / sqrt(var + eps);
                ''',
                'bn_std_inv')(var, self.eps)
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std_inv, T gamma, T beta', 'T x_hat, T y',
                '''
                x_hat = (x - mean) * std_inv;
                y = gamma * x_hat + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std_inv[expander],
                               gamma[expander], beta[expander])
        #
        # end of forward path
        #

        #
        # calculation of lipschitz constant
        #
        if getattr(chainer.config, 'lmt', False):
            # gamma is always positive
            # we took absolute in link/batch_normalization
            tmp_l = gamma * self.std_inv
            self.index = cuda.cupy.argmax(tmp_l)
            l = tmp_l[self.index].reshape((1,))
        else:
            # not used
            l = xp.ones((1,), dtype=xp.float32)

        #
        # calculate running average of statistics
        #
        if configuration.config.train:
            self.running_mean *= self.decay
            self.running_mean += mean * (1 - self.decay)
            self.running_var *= self.decay
            self.running_var += var * (1 - self.decay)
        return y, l

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy, gl = grad_outputs
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)
        if len(inputs) == 5:
            assert not chainer.config.train
            # we do not have to consider Lipschitz constant
            var = inputs[4] + self.eps
            gs = gamma * self.std_inv
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, gmean, gvar

        assert configuration.config.train
        gbeta = gy.sum(axis=axis)
        ggamma = cuda.reduce(
            'T gy, T x_hat', 'T out',
            'gy * x_hat',
            'a + b', 'out = a', '0', 'bn_ggamma')(
            gy, self.x_hat, axis=axis, keepdims=False
        )
        if gl is not None:
            assert getattr(chainer.config, 'lmt', False)
            ggamma[self.index] += gl.reshape(tuple()) * self.std_inv[self.index]
        inv_m = numpy.float32(1) / m
        if xp is numpy:
            gx = (gamma * self.std_inv)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
        else:
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std_inv, T ggamma, T gbeta, \
                T inv_m',
                'T gx',
                'gx = (gamma * std_inv) * (gy - (x_hat * ggamma + gbeta) * \
                inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander],
                          self.std_inv[expander], ggamma[expander],
                          gbeta[expander], inv_m)
        return gx, ggamma, gbeta


def batch_normalization(x, gamma, beta, **kwargs):
    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9))

    return BatchNormalizationFunction(eps, running_mean, running_var,
                                      decay)(x, gamma, beta)


def fixed_batch_normalization(x, gamma, beta, mean, var, eps=2e-5):
    with configuration.using_config('train', False):
        return BatchNormalizationFunction(eps, None, None, 0.0)(
            x, gamma, beta, mean, var)
