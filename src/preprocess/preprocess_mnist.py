import chainer
from chainer.functions import reshape


class PreprocessMNIST(chainer.link.Chain):
    def __init__(self):
        super(PreprocessMNIST, self).__init__()

    def augment(self, x):
        return x

    def __call__(self, x):
        x, t, l = x
        if isinstance(l, int) or isinstance(l, float):
            xp = chainer.cuda.get_array_module(x)
            l = xp.array([l], dtype=xp.float32)
        x = reshape(x, (x.shape[0], 1, 28, 28))
        return x, t, l
