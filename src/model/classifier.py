import chainer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class LMTraining(link.Chain):
    """almost the same with chainer.links.Classifier
    Turn off LMT-mode on validation.

    """

    compute_accuracy = True

    def __init__(self, predictor, preprocess, c=.0, attacker=None,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(LMTraining, self).__init__()
        self.c = c
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.attacker = attacker

        with self.init_scope():
            self.preprocess = preprocess
            self.predictor = predictor

    def __call__(self, *args):

        assert len(args) == 2
        x, t = args
        self.y = None
        self.loss = None
        self.accuracy = None
        if chainer.config.train:
            x = self.preprocess.augment(x)
            if self.attacker is not None:
                x, t = self.attacker(x, t)
            self.y, _, _ = self.predictor(self.preprocess((x, t, self.c)))
        else:
            with chainer.using_config('lmt', False):
                self.y, _, _ = self.predictor(self.preprocess((x, t, self.c)))
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
