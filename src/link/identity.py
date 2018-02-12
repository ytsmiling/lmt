import chainer


class Identity(chainer.Chain):
    """identity mapping
    If you do not want to use batch-normalization layers,
    replace them with this.

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def __call__(self, x):
        return x
