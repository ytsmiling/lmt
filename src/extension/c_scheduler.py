from chainer.training import extension


class CScheduler(extension.Extension):
    """Trainer extension that manages scheduling of the hyperparameter c
    in LMT.

    """

    def __init__(self, func, attr='c', optimizer=None):
        self._attr = attr
        self._func = func
        self._optimizer = optimizer

    def __call__(self, trainer):
        target = self._get_optimizer(trainer).target
        updater = trainer.updater
        val = self._func(getattr(target, self._attr), updater.epoch, updater.epoch_detail, updater.iteration)
        self._update_value(target, val)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, target, value):
        setattr(target, self._attr, value)


class GoyalSchedule:
    """Linearly increase the c from 'start_c' to 'end_c' for every iteration
    in first 'epoch' epoch.

    """

    def __init__(self, start_c, end_c, epoch):
        self.start_c = start_c
        self.end_c = end_c
        self.epoch = epoch

    def __call__(self, c, epoch, epoch_detail, iteration):
        if epoch < self.epoch:
            return (self.end_c * (epoch_detail / self.epoch) +
                    self.start_c * (1 - epoch_detail / self.epoch))
        else:
            return c
