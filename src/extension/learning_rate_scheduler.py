from chainer.training import extension


class LearningRateScheduler(extension.Extension):
    """Trainer extension that manages scheduling of learning rate.

    """

    def __init__(self, func, attr='lr', optimizer=None):
        self._attr = attr
        self._func = func
        self._optimizer = optimizer

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        updater = trainer.updater
        val = self._func(getattr(optimizer, self._attr), updater.epoch, updater.epoch_detail, updater.iteration)
        self._update_value(optimizer, val)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)


class ExponentialSchedule:
    """Multiply 'ratio' to the learning rate after the end of epochs in 'drop_list'.

    """

    def __init__(self, ratio, drop_list):
        self.ratio = ratio
        self.drop_list = drop_list
        self.finished = list()

    def __call__(self, lr, epoch, epoch_detail, iteration):
        if epoch in self.drop_list and epoch not in self.finished:
            lr *= self.ratio
            self.finished.append(epoch)
        return lr
