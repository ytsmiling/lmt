import numpy as np
from chainer.training import extension


def threshold(alpha):
    """Only two-to-one mapping is supported.
    """

    alpha[0] = min(1., max(0., (1. + alpha[0] - alpha[1]) * .5))
    alpha[1] = 1. - alpha[0]


def parseval(W, beta, ratio):
    """Parseval regularization of weight matrices.

    :param W: weight matrices
    :param beta: controls power of the regularization
    :param ratio: ratio of subsampling of rows.
    :return:
    """

    if ratio == 1.:
        w = W.data.reshape((W.shape[0], -1))
        w = (1 + beta) * w - beta * w.dot(w.T).dot(w)
        W.data[:] = w.reshape(W.shape)
    else:
        index = np.random.random(W.shape[0]) < ratio
        w = W.data.reshape((W.shape[0], -1))
        w[index] = (1 + beta) * w[index] - beta * w[index].dot(w[index].T).dot(w[index])
        W.data[:] = w.reshape(W.shape)


class Parseval(extension.Extension):
    """Trainer extension for Parseval networks.
    After each iteration, this extension does
    1. Parseval regularization to weight matrices,
    2. projection to convex polytope of parameters of aggregation layer.

    """

    trigger = 1, 'iteration'
    priority = extension.PRIORITY_READER - 1

    def __init__(self, beta, ratio=1.):
        self.beta = beta
        self.ratio = ratio

    def __call__(self, trainer=None):
        target = trainer.updater.get_optimizer('main').target
        for name, param in target.namedparams():
            if getattr(param, 'parseval_alpha', False):
                threshold(param.data)
            elif param.ndim >= 2 and not getattr(param, 'last_fc', False):
                parseval(param, self.beta, self.ratio)
