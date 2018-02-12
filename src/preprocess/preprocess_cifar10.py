import numpy as np
import chainer


class PreprocessCIFAR10(chainer.link.Chain):
    def __init__(self):
        super(PreprocessCIFAR10, self).__init__()
        with self.init_scope():
            self.mean = np.array([125.30690002, 122.95014954, 113.86599731],
                                 np.float32).reshape((1, 3, 1, 1))
            self.std_inv = 1. / np.array([62.99325562, 62.08860779, 66.70500946],
                                         np.float32).reshape((1, 3, 1, 1))
            self.register_persistent('mean')
            self.register_persistent('std_inv')
        self.l = float(self.std_inv.max())

    def augment(self, x):
        xp = chainer.cuda.get_array_module(x)
        x2 = xp.zeros_like(x, dtype=xp.float32)
        offset_h = np.random.randint(-4, 5, size=(x.shape[0],))
        offset_w = np.random.randint(-4, 5, size=(x.shape[0],))
        for i in range(len(x)):
            x2[i, :, max(offset_h[i], 0):offset_h[i] + 32, max(offset_w[i], 0):offset_w[i] + 32] = (
                x[i, :, max(-offset_h[i], 0):-offset_h[i] + 32, max(-offset_w[i], 0):-offset_w[i] + 32])
            if np.random.randint(0, 2) == 0:
                x2[i] = xp.flip(x2[i], axis=2)
        return x2

    def __call__(self, x):
        x, t, l = x
        x -= self.mean
        x *= self.std_inv
        return x, t, l * self.l
