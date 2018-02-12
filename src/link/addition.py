import numpy as np
import chainer
from src.function.addition import ParsevalAddition


class Addition(chainer.link.Link):
    """Aggregation layer.
    In Parseval networks, convex sum of inputs is calculated.

    """

    def __init__(self):
        super(Addition, self).__init__()
        with self.init_scope():
            self.alpha = chainer.Parameter(np.ones((2,), dtype=np.float32) * .5)
        self.alpha.parseval_alpha = True

    def __call__(self, x0, x1):
        x0, t, l0 = x0
        x1, _, l1 = x1
        if getattr(chainer.config, 'parseval', False):
            x = ParsevalAddition()(x0, x1, self.alpha)
            l = ParsevalAddition()(l0, l1, self.alpha)
        else:
            x = x0 + x1
            l = l0 + l1
        return x, t, l
