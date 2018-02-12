import math
import chainer
from chainer import functions as F
from src.attack.attacker_base import AttackerBase
from src.attack.consistency import projection_l2


class PGDL2(AttackerBase):
    """PGD with l_2 constraint.
    I'm not sure whether this is a right code for PGD in l_2-ball.
    At least, this is weak compared to the original PGD,
    which performs gradient descent in l_\infty-ball.

    """

    def __init__(self, target, epsilon, lr, n_step, n_restart=1):
        super(PGDL2, self).__init__(target)
        self.epsilon = epsilon
        self.lr = lr
        self.n_step = n_step
        self.n_restart = n_restart

    def craft(self, image, label):
        original = image
        max_loss = -1.
        best_image = None
        for _ in range(self.n_restart):
            image = original.copy()
            xp = chainer.cuda.get_array_module(image)
            image += (xp.random.random(image.shape) - 0.5) * 2 * self.epsilon / math.sqrt(image.size)
            image = chainer.Parameter(image)
            for i in range(self.n_step):
                prediction = self.predict(image, label, backprop=True)
                image.grad = None
                with chainer.force_backprop_mode():
                    self.backprop(loss=F.softmax_cross_entropy(prediction, label))
                image.data += self.lr * chainer.cuda.cupy.sign(image.grad)
                projection_l2(image, original, self.epsilon)
            if self.n_restart > 1:
                prediction = self.predict(image, label, backprop=False)
                loss = F.softmax_cross_entropy(prediction, label)
                assert loss.data > 0
                if max_loss < loss.data:
                    max_loss = loss.data
                    best_image = image.data
            else:
                best_image = image.data
        return best_image
