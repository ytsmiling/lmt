import chainer
import abc


class AttackerBase(chainer.Link):
    def __init__(self, model):
        super(AttackerBase, self).__init__()
        with self.init_scope():
            self.model = model
        self.l2_history = list()

    def register_model(self, model):
        with self.init_scope():
            self.model = model

    def __call__(self, image, label):
        image = self.craft(image, label)
        return image, label

    def predict(self, image, label, backprop=True):
        with chainer.using_config('train', False):
            if backprop:
                with chainer.force_backprop_mode():
                    ret = self.model.predictor(self.model.preprocess((image, label, 0.)))
            else:
                with chainer.no_backprop_mode():
                    ret = self.model.predictor(self.model.preprocess((image, label, 0.)))
            return ret[0]

    def backprop(self, loss):
        use_cleargrads = getattr(self.model, '_use_cleargrads', True)
        if use_cleargrads:
            self.model.cleargrads()
        else:
            self.model.zerograds()
        loss.backward()
        if use_cleargrads:
            self.model.cleargrads()
        else:
            self.model.zerograds()

    @abc.abstractmethod
    def craft(self, image, label):
        return image, label
