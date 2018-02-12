import chainer
from src.function.reshape import flatten
from src.function.activation import relu
from src.link.convolution_2d import Convolution2D
from src.link.linear import Linear
from src.link.last_linear import LastLinear


class OuterPolytope(chainer.Chain):
    """Convolutional network with ReLU.
    We followed [Provable defenses against adversarial examples via the convex outer adversarial polytope]
    (https://arxiv.org/abs/1711.00851). This network follows their public code.

    conv-4x4 channel 16, stride 2x2, pad 1x1
    ReLU
    conv-4x4 channel 32, stride 2x2, pad 1x1
    ReLU
    fc 100
    ReLU
    fc 10
    """

    def __init__(self):
        super(OuterPolytope, self).__init__()
        initialW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = Convolution2D(in_channels=1, out_channels=16, ksize=4, stride=2,
                                       pad=1, initialW=initialW, nobias=False)
            self.conv2 = Convolution2D(in_channels=16, out_channels=32, ksize=4, stride=2,
                                       pad=1, initialW=initialW, nobias=False)
            self.fc1 = Linear(100)
            self.fc2 = LastLinear(10)

    def __call__(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = flatten(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x
