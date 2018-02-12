import chainer
from src.function.activation import relu
from src.function.reshape import flatten
from src.function.pooling import max_pooling_2d
from src.link.convolution_2d import Convolution2D
from src.link.linear import Linear
from src.link.last_linear import LastLinear


class LeNet(chainer.Chain):
    """LeNet-like architecture.
    We followed [Lower bounds on the robustness to adversarial perturbations]
    (https://papers.nips.cc/paper/6682-lower-bounds-on-the-robustness-to-adversarial-perturbations).

    conv-5x5 channel 20, stride 1x1
    max_pool-2x2 stride 2x2
    conv-5x5 channel 50, stride 1x1
    max_pool-2x2 stride 2x2
    fc 500
    activation
    fc 10
    """

    def __init__(self):
        super(LeNet, self).__init__()
        initialW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = Convolution2D(in_channels=None, out_channels=20, ksize=5, stride=1,
                                       pad=0, initialW=initialW, nobias=False)
            self.conv2 = Convolution2D(in_channels=20, out_channels=50, ksize=5, stride=1,
                                       pad=0, initialW=initialW, nobias=False)
            self.fc1 = Linear(500)
            self.fc2 = LastLinear(10)

    def __call__(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = max_pooling_2d(x, ksize=2, stride=2, pad=0)
        x = self.conv2(x)
        x = relu(x)
        x = max_pooling_2d(x, ksize=2, stride=2, pad=0)
        x = flatten(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
