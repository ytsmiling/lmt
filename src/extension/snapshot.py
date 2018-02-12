import os
import chainer
from chainer.training import extension


class Snapshot(extension.Extension):
    """Trainer extension that save network parameters to 'snapshot.npz'.

    """
    def __call__(self, trainer):
        filename = os.path.join(trainer.out, 'snapshot.npz')
        chainer.serializers.save_npz(filename, trainer.updater.get_optimizer('main').target)
