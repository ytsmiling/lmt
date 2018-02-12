import chainer


class ParsevalAddition(chainer.function.Function):
    """Implementation of aggregation layer for Parseval networks.
    Only two to one mapping is supported.

    """

    def forward(self, inputs):
        x0, x1, alpha = inputs
        return x0 * alpha[0] + x1 * alpha[1],

    def backward(self, inputs, grad_outputs):
        x0, x1, alpha = inputs
        gy = grad_outputs[0]
        xp = chainer.cuda.get_array_module(gy)
        ga = xp.array([(gy * x0).sum(), (gy * x1).sum()], xp.float32)
        return gy * alpha[0], gy * alpha[1], ga
