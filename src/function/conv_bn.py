import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import argument

from src.function.normalize import normalize


class BatchNormalizationFunction(function.Function):
    """Calculates batch normalization.
    This function also receives weight matrix (tensor?) of
    leading layer (fully-connected or convolutional layer)
    If in LMT mode, this also calculates approximation of the Lipschitz constant of
    the combined layer ((fc + bn) or (conv + bn)) using the power iteration method
    """

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9, u=None):
        self.running_mean = mean
        self.running_var = var
        self.eps = eps
        self.decay = decay
        self.u = u

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
        elif len(inputs) == 6:
            self.fixed_mean = inputs[4]
            self.fixed_var = inputs[5]

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
                'a + b', 'out = alpha * a', '0', 'conv_bn_var')(
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
                'conv_bn_std_inv')(var, self.eps)
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std_inv, T gamma, T beta', 'T x_hat, T y',
                '''
                x_hat = (x - mean) * std_inv;
                y = gamma * x_hat + beta;
                ''',
                'conv_bn_fwd')(x, mean[expander], self.std_inv[expander],
                               gamma[expander], beta[expander])
        #
        # end of forward path
        #

        #
        # calculation of lipschitz constant
        #
        if chainer.config.train and getattr(chainer.config, 'lmt', False):
            #
            # power iteration for a matrix Diag(\gamma_i/\sigma_i)W
            #
            # u <= Diag(\gamma_i/\sigma_i) u
            # v <= W u
            # u_mid <= W^T v
            # u <= Diag(\gamma_i/\sigma_i)^T v
            #
            W = inputs[3].reshape((inputs[3].shape[0], -1))

            tmp_l = gamma * self.std_inv
            self.u *= tmp_l
            self.v = self.u.dot(W)

            # normalize for back propagation
            normalize(self.v, eps=1e-20)

            # do not normalize u_mid
            self.u_mid = self.v.dot(W.T)

            self.u[:] = self.u_mid * tmp_l

            # normalize for back propagation
            nu = normalize(self.u, eps=1e-20)

            # spectral norm is approximated by the norm of a vector u
            l = nu.reshape((1,))
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
        #
        # preprocess
        #
        x, gamma = inputs[:2]
        gy, gl = grad_outputs
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)

        if len(inputs) == 6:
            assert not chainer.config.train
            # we do not have to consider Lipschitz constant
            var = inputs[5] + self.eps
            gs = gamma * self.std_inv
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy
            return gx, ggamma, gbeta, None, gmean, gvar

        assert chainer.config.train
        gbeta = gy.sum(axis=axis)
        ggamma = cuda.reduce(
            'T gy, T x_hat', 'T out',
            'gy * x_hat',
            'a + b', 'out = a', '0', 'conv_bn_ggamma')(
            gy, self.x_hat, axis=axis, keepdims=False
        )
        if gl is not None:
            assert getattr(chainer.config, 'lmt', False)
            cuda.elementwise(
                'T gl, T u, T u_mid, T std_inv', 'T ggamma',
                '''
                ggamma += gl * u * u_mid * std_inv;
                ''',
                'conv_bn_ggamma2')(gl, self.u.reshape(self.std_inv.shape),
                                   self.u_mid.reshape(self.std_inv.shape),
                                   self.std_inv, ggamma)
        inv_m = numpy.float32(1) / m
        if xp is numpy:
            gx = (gamma * self.std_inv)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
        else:
            # in LMT, ggamma is changed and this automatically corrects gx
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std_inv, T ggamma, T gbeta, \
                T inv_m',
                'T gx',
                'gx = (gamma * std_inv) * (gy - (x_hat * ggamma + gbeta) * \
                inv_m)',
                'conv_bn_bwd')(gy, self.x_hat, gamma[expander],
                               self.std_inv[expander], ggamma[expander],
                               gbeta[expander], inv_m)
        if gl is not None:
            return gx, ggamma, gbeta, (gl * self.u_mid.T * self.v).reshape(inputs[3].shape)
        else:
            return gx, ggamma, gbeta, None


def batch_normalization(x, gamma, beta, W, u, **kwargs):
    eps, running_mean, running_var, decay = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9))

    return BatchNormalizationFunction(eps, running_mean, running_var,
                                      decay, u=u)(x, gamma, beta, W)


def fixed_batch_normalization(x, gamma, beta, mean, var, eps, W, u):
    with configuration.using_config('train', False):
        return BatchNormalizationFunction(eps, None, None, 0.0, u=u)(
            x, gamma, beta, W, mean, var)
