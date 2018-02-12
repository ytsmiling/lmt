import copy
import os

import numpy as np
import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer import function
from chainer import reporter as reporter_module
from chainer.training import extension


class Margin(extension.Extension):
    """This class is an extension of Trainer module.
    This calculates invariant radii of target for given dataset.
    """

    trigger = 1, 'epoch'
    default_name = 'margin'
    priority = extension.PRIORITY_WRITER - 1

    def __init__(self, dataset, target, preprocess, batchsize, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None):
        self.iterator = chainer.iterators.SerialIterator(
            dataset, batchsize, repeat=False, shuffle=False)

        self.target = target
        self.preprocess = preprocess

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

    def __call__(self, trainer=None):
        # set up a reporter
        reporter = reporter_module.Reporter()
        reporter.add_observer(self.name, self.target)

        with reporter:
            with configuration.using_config('train', False):
                with configuration.using_config('lmt', True):
                    with configuration.using_config('lmt-fc', True):
                        with configuration.using_config('exact', True):
                            with configuration.using_config('cudnn_deterministic', True):
                                self.evaluate(os.path.join(trainer.out, 'margin.npy'))

    def evaluate(self, output_file_name):
        iterator = self.iterator
        preprocess = self.preprocess
        target = self.target
        eval_func = self.eval_func or (lambda x: target(preprocess(x)))
        device = self.device or chainer.cuda.cupy.cuda.get_device_id()

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        margin_list = []
        for batch in it:
            in_arrays = self.converter(batch, device)
            with function.no_backprop_mode():
                assert isinstance(in_arrays, tuple)
                xp = chainer.cuda.get_array_module(*in_arrays)
                y, t, lipschitz = eval_func((in_arrays + (xp.ones((1,), dtype=xp.float32),)))
                y = y.data
                lipschitz = lipschitz.data
                assert y.size == lipschitz.size
                assert y.ndim == 2
                # calculate Lipschitz-normalized margin
                margins = chainer.cuda.elementwise(
                    'T y, T yt, T lipschitz',
                    'T margin',
                    '''
                    if (lipschitz == 0.0) {
                      // margin is INF
                      // e.g. y == yt
                      margin = 255.0 * 3.0 * 300 * 300;
                    } else {
                      margin = (yt - y) / lipschitz;
                      if (margin < 0.0) {
                        margin = 0.0;
                      }
                    }
                    ''',
                    'margin'
                )(y, y[list(range(t.size)), t].reshape(t.size, 1), lipschitz)
                margins = xp.min(margins, axis=1)
                margin_list.extend(list(margins.get()))

        margin_list = np.asarray(margin_list)
        np.save(output_file_name, margin_list)
