import math
import numpy as np
import chainer


class LmtFc(chainer.function.Function):
    """Calculates Lipschitz constant for the last fully-connected layer.
    This also apply addition to its input for LMT++.

    """

    def __init__(self, t):
        self.t = t

    def forward(self, inputs):
        x, W, l = inputs
        assert l.shape == (1,), 'l.shape == {}'.format(l.shape)
        t = self.t
        xp = chainer.cuda.get_array_module(W)

        # this calculates w_i - w_t for each batch and each row
        self.diff = W.reshape((1,) + W.shape) - W[t].reshape((t.size, 1, W.shape[1]))
        assert self.diff.shape == (x.shape[0], W.shape[0], W.shape[1])

        # this calculates \|w_i - w_t\|_2 for each batch and each row
        w_norm = xp.linalg.norm(self.diff, axis=2)
        assert w_norm.shape == (x.shape[0], W.shape[0])  # (batch size, class size)

        # for back propagation
        self.diff /= w_norm.reshape(w_norm.shape + (1,)) + 1e-22
        self.w_norm = w_norm

        l = l * self.w_norm
        if xp == np:
            xt = x[list(range(t.size)), t].reshape((t.size, 1))
            pred_margin = xt - x
            factor = np.where(pred_margin >= 0,
                              np.where(pred_margin >= l, 1, pred_margin / (l + 1e-20)),
                              0)
            self.factor = np.amin(factor, axis=1, keepdims=True)
            y = x + factor * l
        else:
            # we need output corresponding to the target class
            # because we scale addition corresponding to the output margin
            xt = x[list(range(t.size)), t].reshape((t.size, 1))
            factor = chainer.cuda.cupy.ElementwiseKernel(
                'T x, T xt, T l',
                'T factor',
                """
                float prediction_margin = xt - x;
                if (prediction_margin >= 0) {
                  factor = prediction_margin >= l ? 1 : (prediction_margin / l);
                } else {
                  factor = 0;
                }
                """,
                'last_fc_forward_1'
            )(x, xt, l)
            self.factor = chainer.cuda.cupy.amin(factor, axis=1, keepdims=True)
            y = chainer.cuda.cupy.ElementwiseKernel(
                'T x, T l, T factor',
                'T y',
                """
                y = x + factor * l;
                """,
                'last_fc_forward_2'
            )(x, l, self.factor)
            # y = chainer.cuda.cupy.ElementwiseKernel(
            #     'T x, T l',
            #     'T y',
            #     """
            #     y = x + l;
            #     """,
            #     'last_fc_forward_2'
            # )(x, l)
        return y, l

    def backward(self, inputs, grad_outputs):
        x, W, l = inputs
        t = self.t
        gy, gl = grad_outputs
        if gy is not None:
            assert gl is None
            assert gy.shape == (x.shape[0], W.shape[0])
            assert self.factor.shape == (x.shape[0], 1)
            gy = gy * self.factor
            gy2 = gy * l
            assert gy.shape == (x.shape[0], W.shape[0])
            xp = chainer.cuda.get_array_module(W)
            assert self.diff.shape == (x.shape[0], W.shape[0], W.shape[1])
            self.diff *= gy2.reshape(gy2.shape + (1,))
            assert self.diff.shape == (x.shape[0], W.shape[0], W.shape[1])
            ret = self.diff
            ret[list(range(t.size)), t] -= xp.sum(self.diff, axis=1)
            assert ret.shape == (x.shape[0], W.shape[0], W.shape[1])
            ret = ret.sum(0)
            assert ret.shape == (W.shape[0], W.shape[1])
            return grad_outputs[0], ret, (gy * self.w_norm).sum().reshape((1,))
        else:
            assert gy is None
            gl = grad_outputs[1]
            gl = gl * self.factor
            return None, (gl.reshape(gl.shape + (1,)) * self.diff).sum(0), (gl * self.w_norm).sum().reshape((1,))


class Lmt(chainer.function.Function):
    def __init__(self, t):
        self.t = t

    def forward(self, inputs):
        x, l = inputs
        assert l.shape == (1,)
        l = l * math.sqrt(2)
        t = self.t
        xp = chainer.cuda.get_array_module(x)
        x2 = x.copy()
        x2[list(range(t.size)), t] = xp.min(x, 1)
        self.factor = xp.clip((x[list(range(t.size)), t] - xp.max(x2, 1)) / l, 0., 1.)
        x2[list(range(t.size)), t] = x[list(range(t.size)), t] - self.factor * l
        return x2, l

    def backward(self, inputs, grad_outputs):
        gy, gl = grad_outputs
        assert gl is None
        return gy, -(gy[list(range(self.t.size)), self.t] * self.factor).sum().reshape((1,)) * math.sqrt(2)
