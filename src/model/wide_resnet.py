import chainer
from chainer import initializers
from src.function.activation import relu
from src.function.dropout import dropout
from src.function.pooling import average_pooling_2d
from src.link.addition import Addition
from src.link.convolution_2d import Convolution2D
from src.link.last_linear import LastLinear
from src.link.batch_normalization import BatchNormalization


class BasicBlock(chainer.Chain):
    def __init__(self, in_channel, out_channel, stride, reduce, drop, normalization, nobias):
        super(BasicBlock, self).__init__()
        initialW = initializers.HeNormal()

        self.reduce = reduce
        self.drop = drop
        with self.init_scope():
            self.shortcut = Convolution2D(
                in_channel, out_channel, 1, stride, 0, initialW=initialW, nobias=nobias
            ) if reduce else (lambda x: x)
            self.conv1 = Convolution2D(
                in_channel, out_channel, 3, stride, 1, initialW=initialW, nobias=nobias)
            self.conv2 = Convolution2D(
                out_channel, out_channel, 3, 1, 1, initialW=initialW, nobias=nobias)
            self.bn1 = normalization(in_channel)
            self.bn2 = normalization(out_channel)
            self.addition = Addition()

    def __call__(self, x):
        h = relu(self.bn1(x))
        if self.reduce:
            x = h
        h = dropout(relu(self.bn2(self.conv1(h))), ratio=self.drop)
        h = self.conv2(h)
        h_s = self.shortcut(x)
        return self.addition(h, h_s)


class Block(chainer.Chain):
    def __init__(self, n, in_channel, out_channel, stride=2, drop=.3, normalization=None, nobias=True):
        super(Block, self).__init__()
        self.n = n
        with self.init_scope():
            reduce = True
            for i in range(n):
                setattr(self, 'block{}'.format(i), BasicBlock(in_channel, out_channel, stride,
                                                              reduce, drop, normalization, nobias))
                reduce = False
                stride = 1
                in_channel = out_channel

    def __call__(self, x):
        for i in range(self.n):
            x = getattr(self, 'block{}'.format(i))(x)
        return x


class WideResNet(chainer.Chain):
    """Wide Residual Network.
    see [Wide Residual Networks](https://arxiv.org/abs/1605.07146)

    """

    def __init__(self, k, n_layer, drop, normalization=BatchNormalization, n_class=10, nobias=True):
        """

        :param k: width factor
        :param n_layer: depth of the network, this must be 6n + 4
        :param drop: dropout ratio
        :param normalization: normalization method (e.g. BatchNormalization)
        """
        super(WideResNet, self).__init__()
        if n_layer < 10 or (n_layer - 4) % 6:
            raise ValueError('n_layer must be 6n + 4 for a some positive integer n.')
        n = (n_layer - 4) // 6
        with self.init_scope():
            self.conv1 = Convolution2D(
                3, 16, 3, 1, 1, initialW=initializers.HeNormal(), nobias=nobias)
            self.conv2 = Block(n, 16, 16 * k, stride=1, drop=drop, normalization=normalization, nobias=nobias)
            self.conv3 = Block(n, 16 * k, 32 * k, drop=drop, normalization=normalization, nobias=nobias)
            self.conv4 = Block(n, 32 * k, 64 * k, drop=drop, normalization=normalization, nobias=nobias)
            self.bn = normalization(64 * k)
            self.fc = LastLinear(64 * k, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = relu(self.bn(h))
        h = average_pooling_2d(h, 8, stride=1)
        return self.fc(h)

