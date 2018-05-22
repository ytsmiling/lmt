import chainer


class PreprocessSVHN(chainer.link.Chain):
    def __init__(self):
        super(PreprocessSVHN, self).__init__()

    def augment(self, x):
        return x

    def __call__(self, x):
        x, t, l = x
        if isinstance(l, int) or isinstance(l, float):
            xp = chainer.cuda.get_array_module(x)
            l = xp.array([l], dtype=xp.float32)
        return x, t, l
