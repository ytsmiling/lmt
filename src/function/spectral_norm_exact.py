import chainer


def spectral_norm_exact(W):
    """This calculates the spectral norm using SVD.

    :param W: numpy or cupy matrix
    :return: spectral norm of W
    """
    # numpy or cupy (GPU implementation of numpy) is automatically selected
    xp = chainer.cuda.get_array_module(W)

    # It seems cupy.linalg.svd(W) do not ensure that
    # W is immutable, so we copy it
    w = W.copy().reshape((W.shape[0], -1))

    return max(abs(xp.linalg.svd(w)[1]))
