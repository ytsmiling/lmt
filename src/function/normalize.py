from chainer import cuda


def normalize(arr, eps):
    """normalize input array and return its norm
    from https://github.com/pfnet-research/sngan_projection/blob/master/source/functions/max_sv.py#L5

    :param arr: numpy ndarray or cupy ndarray
    :param eps: epsilon for numerical stability
    :return: norm of input array
    """

    norm = cuda.reduce('T x', 'T out',
                       'x * x', 'a + b', 'out = sqrt(a)', 0,
                       'norm_sn')(arr)
    cuda.elementwise('T norm, T eps',
                     'T x',
                     'x /= (norm + eps)',
                     'div_sn')(norm, eps, arr)
    return norm
