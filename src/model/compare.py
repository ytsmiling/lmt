import chainer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class Compare(link.Chain):
    """Classifier for the Lipschitz constant regularization.
    You can also test mere addition of constant to outputs (
    set chainer.config.constant_addition = True for it).

    """

    compute_accuracy = True

    def __init__(self, predictor, preprocess, c=.0,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Compare, self).__init__()
        self.c = c
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.preprocess = preprocess
            self.predictor = predictor

    def __call__(self, *args):

        assert len(args) == 2
        x = args[0]
        t = args[1]
        self.y = None
        self.loss = None
        self.accuracy = None
        if chainer.config.train:
            x = self.preprocess.augment(x)
            if getattr(chainer.config, 'lipschitz_regularization', False):
                self.y, _, L = self.predictor(self.preprocess((x, t, 1.)))
            else:
                self.y, _, L = self.predictor(self.preprocess((x, t, self.c)))
            if getattr(chainer.config, 'constant_addition', False):
                self.y.data += self.c
                self.y.data[list(range(t.size)), t] -= self.c
        else:
            self.y, _, L = self.predictor(self.preprocess((x, t, self.c)))
        self.loss = self.lossfun(self.y, t)
        if getattr(chainer.config, 'lipschitz_regularization', False):
            self.loss += chainer.functions.sum(L ** 2) / 2 * self.c
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
