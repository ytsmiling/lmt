import numpy as np


def orthonormal(W):
    """Orthonormal initialization.
    Note: Parseval networks just used HeNormal for their initialization and\
    this function was not used in our experiments.

    :param W (chainer.Parameter): weight matrix
    :return: normalized weight matrix
    """
    w = W.data.reshape((W.shape[0], -1))
    s, _, d = np.linalg.svd(w)
    v = np.zeros((s.shape[0], d.shape[0]), dtype=np.float32)
    np.fill_diagonal(v, 1)
    W.data[:] = s.dot(v).dot(d).reshape(W.shape)
