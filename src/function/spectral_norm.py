import chainer

from src.function.normalize import normalize


class SpectralNormFunction(chainer.function.Function):
    """Calculates spectral norm of weight matrix in differentiable way
    using power iteration method.
    """

    def __init__(self, u):
        """Let W be a target weight matrix with shape (h, w).

        :param u (cupy.ndarray): a vector with shape (1, h) for the power
        iteration method.
        """
        self.u = u
        self.v = None

    def forward(self, inputs):
        W = inputs[0].reshape((inputs[0].shape[0], -1))

        # power iteration
        self.v = self.u.dot(W)
        normalize(self.v, eps=1e-20)

        self.u[:] = self.v.dot(W.T)
        nu = normalize(self.u, eps=1e-20)

        # spectral norm is approximated by the norm of a vector u
        return nu.reshape((1,)),

    def backward(self, inputs, grad_outputs):
        assert grad_outputs[0].shape == (1,)
        # \frac{\partial spectral_norm}{\partial W}
        # = \frac{\partial u^TWv}{\partial W}
        # = uv^T
        #
        # note: u and v are normalized vectors
        return (grad_outputs[0] * self.u.T * self.v).reshape(inputs[0].shape),
