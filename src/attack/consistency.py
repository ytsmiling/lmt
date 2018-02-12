import chainer


def projection(image, original, epsilon):
    """Projection to l_\infty ball.
    """

    xp = chainer.cuda.get_array_module(original)
    image.data = xp.clip(image.data, original - epsilon, original + epsilon)
    image.data = xp.clip(image.data, 0, 255)


def projection_l2(image, original, epsilon):
    """Projection to l_2 ball.
    """

    xp = chainer.cuda.get_array_module(original)
    diff = image.data - original
    norm = xp.linalg.norm(diff)
    if norm > epsilon:
        image.data[:] = original + diff / norm * epsilon
    image.data = xp.clip(image.data, 0, 255)
