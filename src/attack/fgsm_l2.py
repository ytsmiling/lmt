import chainer
from chainer import functions as F
from src.attack.attacker_base import AttackerBase


class FGSML2(AttackerBase):
    """More precisely, one-step attack in l_2-ball.

    """

    def __init__(self, model, epsilon):
        super(FGSML2, self).__init__(model)
        self._epsilon = epsilon

    def craft(self, image, label):
        image = chainer.Parameter(image)
        prediction = self.predict(image, label, backprop=True)
        with chainer.force_backprop_mode():
            self.backprop(loss=F.softmax_cross_entropy(prediction, label))
        xp = chainer.cuda.get_array_module(image.grad)
        image = image.data + self._epsilon / xp.linalg.norm(image.grad) * image.grad
        return image
