import chainer


def dropout(x, ratio=.5, **kwargs):
    """dropout regularization
    Even though it scales its input at training,
    we do not consider it in Lipschitz constant.

    :param x: input (vector/tensor, label, lipschitz)
    :param ratio: dropout ratio
    :return:
    """
    x, t, l = x
    x = chainer.functions.dropout(x, ratio=ratio, **kwargs)
    return x, t, l
