import chainer
from chainer import initializers
from src.function.activation import relu
from src.function.dropout import dropout
from src.function.pooling import average_pooling_2d
from src.link.addition import Addition
from src.link.convolution_2d import Convolution2D
from src.link.last_linear import LastLinear
from src.link.batch_normalization import BatchNormalization
from src.link.conv_bn import Convolution2DBN


class BasicBlock(chainer.Chain):
    def __init__(self, in_channel, out_channel, stride, reduce, drop):
        super(BasicBlock, self).__init__()
        initialW = initializers.HeNormal()

        self.reduce = reduce
        self.drop = drop
        self.stride = stride
        with self.init_scope():
            if self.reduce:
                self.shortcut = Convolution2D(
                    in_channel, out_channel, 1, 1, 0, initialW=initialW,
                    nobias=True
                )
            self.conv1 = Convolution2DBN(
                in_channel, out_channel, 3, stride, 1, initialW=initialW,
                nobias=True)
            self.conv2 = Convolution2D(
                out_channel, out_channel, 3, 1, 1, initialW=initialW,
                nobias=True)
            self.bn = BatchNormalization(out_channel)
            self.addition = Addition()

    def __call__(self, x, bnx):
        if self.reduce:
            if self.stride > 1:
                assert self.stride == 2
                h_s = average_pooling_2d(bnx, ksize=2, stride=2, pad=0)
            else:
                h_s = bnx
            h_s = self.shortcut(h_s)
        else:
            h_s = x
        h = dropout(relu(self.conv1(bnx)), ratio=self.drop)
        h = self.conv2(h)
        h = self.addition(h, h_s)
        return h, relu(self.bn(h))


class Block(chainer.Chain):
    def __init__(self, n, in_channel, out_channel, stride=2, drop=.3):
        super(Block, self).__init__()
        self.n = n
        with self.init_scope():
            reduce = True
            for i in range(n):
                setattr(self, 'block{}'.format(i),
                        BasicBlock(in_channel, out_channel, stride,
                                   reduce, drop))
                reduce = False
                stride = 1
                in_channel = out_channel

    def __call__(self, x, bnx):
        for i in range(self.n):
            x, bnx = getattr(self, 'block{}'.format(i))(x, bnx)
        return x, bnx


class WideResNet(chainer.Chain):
    """Wide Residual Network.
    see [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
    Upper bound of the Lipschitz constant of convolutional layer + batch-normalization layer
    in residual blocks are jointly calculated.

    """

    def __init__(self, k, n_layer, drop, n_class=10):
        """

        :param k: width factor
        :param n_layer: depth of the network, this must be 6n + 4
        :param drop: dropout ratio
        :param normalization: normalization method (e.g. BatchNormalization)
        """
        super(WideResNet, self).__init__()
        if n_layer < 10 or (n_layer - 4) % 6:
            raise ValueError(
                'n_layer must be 6n + 4 for a some positive integer n.')
        n = (n_layer - 4) // 6
        with self.init_scope():
            self.conv1 = Convolution2DBN(
                3, 16, 3, 1, 1, initialW=initializers.HeNormal(), nobias=True)
            self.conv2 = Block(n, 16, 16 * k, stride=1, drop=drop)
            self.conv3 = Block(n, 16 * k, 32 * k, drop=drop)
            self.conv4 = Block(n, 32 * k, 64 * k, drop=drop)
            self.fc = LastLinear(64 * k, n_class)

    def __call__(self, x):
        bnh = self.conv1(x)
        h, bnh = self.conv2(None, bnh)
        h, bnh = self.conv3(h, bnh)
        h, bnh = self.conv4(h, bnh)
        h = average_pooling_2d(bnh, 8, stride=1)
        return self.fc(h)
