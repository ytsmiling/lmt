import chainer
from chainer.functions import reshape


class PreprocessMNIST(chainer.link.Chain):
    def __init__(self):
        super(PreprocessMNIST, self).__init__()

    def augment(self, x):
        return x

    def __call__(self, x):
        x, t, l = x
        x = reshape(x, (x.shape[0], 1, 28, 28))
        return x / 255, t, l / 255
