import numpy as np
from chainer.backends import cuda

if cuda.available:
    def normalize(arr):
        """normalize input array and return its norm
        from https://github.com/pfnet-research/sngan_projection/blob/master/source/functions/max_sv.py#L5

        :param arr: numpy ndarray or cupy ndarray
        :return: norm of input array
        """

        norm = cuda.reduce('T x', 'T out',
                           'x * x', 'a + b', 'out = sqrt(a)', 0,
                           'norm_sn')(arr)
        cuda.elementwise('T norm',
                         'T x',
                         'x /= (norm + 1e-20)',
                         'div_sn')(norm, arr)
        return norm
else:
    def normalize(arr):
        norm = np.linalg.norm(arr)
        arr /= norm + 1e-20
        return norm
