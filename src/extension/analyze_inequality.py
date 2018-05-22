import copy
import pathlib
import sys

import numpy as np
import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer import function
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer import functions as F

from src.function.normalize import normalize


class AnalyzeInequality(extension.Extension):
    trigger = 1, 'epoch'
    default_name = 'analyze_inequality'
    name = 'analyze_inequality'
    priority = extension.PRIORITY_WRITER - 1

    get_margin = chainer.cuda.elementwise(
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
        'lipschitz_margin'
    )

    def __init__(self, iter, target, preprocess, n_class, attack,
                 output_dir='', converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, nograd=False,
                 attack_name=''):
        self.iterator = iter

        self.target = target.predictor
        self.preprocess = preprocess
        self.n_class = n_class
        self.attack = attack
        self.output_dir = output_dir

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func
        self.nograd = nograd
        self.attack_name = attack_name

        self.global_lipschitz = None

    def __call__(self, trainer=None):
        # set up a reporter
        reporter = reporter_module.Reporter()
        reporter.add_observer(self.name, self.target)

        with reporter:
            with configuration.using_config('cudnn_deterministic', True):
                with configuration.using_config('train', False):
                    with configuration.using_config('lmt', True):
                        with configuration.using_config('lmt-fc', True):
                            with configuration.using_config('exact', True):
                                upper = self.calculate_upper_lipschitz()
                    with configuration.using_config('lmt', False):
                        with configuration.using_config('lmt-fc', False):
                            with configuration.using_config('exact', False):
                                if not self.nograd:
                                    loc = self.calculate_local_lipschitz()
                                    glo = self.calculate_global_lipschitz()
                                adv = self.calculate_adversarial_perturbation()
        print('\revaluation end, saving result', flush=True)
        if self.nograd:
            values = np.array(list(zip(upper, adv)))
        else:
            values = np.array(list(zip(upper, glo, loc, adv)))
        output_dir = self.output_dir or trainer.out
        filename = pathlib.Path(output_dir) / 'inequlaity_{0}.npy'.format(
            self.attack_name)
        np.save(str(filename), values)
        print('\rassertions start', flush=True)
        if self.nograd:
            for up, ad in zip(upper, adv):
                assert up <= ad
        else:
            for up, gl, lo, ad in zip(upper, glo, loc, adv):
                assert up <= gl
                assert gl <= lo
                assert up <= ad

    def calculate_upper_lipschitz(self):
        print('\rupper bound of Lipschitz start', flush=True)
        iterator = self.iterator
        preprocess = self.preprocess
        target = self.target
        eval_func = self.eval_func or (lambda x: target(preprocess(x)))
        device = self.device or chainer.cuda.cupy.cuda.get_device_id()
        assert device >= 0

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
                y, t, lipschitz = eval_func(
                    (in_arrays + (xp.ones((1,), dtype=xp.float32),)))
                y = y.data
                lipschitz = lipschitz.data
                assert y.size == lipschitz.size
                assert y.ndim == 2
                # calculate Lipschitz-normalized margin
                margins = self.get_margin(
                    y, y[list(range(t.size)), t].reshape(t.size, 1), lipschitz)
                margins = xp.min(margins, axis=1)
                margin_list.extend(list(margins.get()))

        return margin_list

    def calculate_local_lipschitz(self):
        print('\rlocal Lipschitz start', flush=True)
        iterator = self.iterator
        preprocess = self.preprocess
        target = self.target
        eval_func = self.eval_func or (lambda x: target(preprocess(x)))
        device = self.device or chainer.cuda.cupy.cuda.get_device_id()
        assert device >= 0

        if self.eval_hook:
            self.eval_hook(self)

        # gradを計算して勾配をsamplingする
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        self.global_grad = chainer.cuda.cupy.zeros(
            (self.n_class, self.n_class), dtype=chainer.cuda.cupy.float32)

        margin_list = []
        size = 0
        total = len(it.dataset)
        for batch in it:
            size += len(batch)
            sys.stdout.write('\r{0}/{1}'.format(size, total))
            sys.stdout.flush()
            x, t = self.converter(batch, device)
            xp = chainer.cuda.get_array_module(x)
            c = xp.ones((1,), dtype=np.float32)
            local_grad = xp.zeros(
                (self.n_class, self.n_class), dtype=xp.float32)
            with chainer.force_backprop_mode():
                for _ in range(100):
                    noise = xp.random.normal(size=x.shape).astype(xp.float32)
                    normalize(noise)
                    x2 = chainer.Parameter(x + noise)
                    y, t, _ = eval_func((x2, t, c))
                    for i in range(self.n_class):
                        for j in range(i + 1, self.n_class):
                            if i == j:
                                continue
                            target.cleargrads()
                            x2.grad = None
                            F.sum(y[:, i] - y[:, j]).backward()
                            norm = xp.max(xp.sqrt((x2.grad ** 2).sum(
                                axis=tuple(range(1, x2.ndim)))))
                            local_grad[i, j] = max(local_grad[i, j], norm)
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    local_grad[j, i] = local_grad[i, j]
                    self.global_grad[:] = xp.maximum(
                        self.global_grad, local_grad)
            with chainer.no_backprop_mode():
                y, t, _ = eval_func((x, t, c))
                y = y.array
            grad = local_grad[t]
            margins = self.get_margin(
                y, y[list(range(t.size)), t].reshape(t.size, 1), grad)
            margins = xp.min(margins, axis=1)
            margin_list.extend(list(margins.get()))

        return margin_list

    def calculate_global_lipschitz(self):
        print('\rglobal Lipschitz start', flush=True)
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
            x, t = self.converter(batch, device)
            xp = chainer.cuda.get_array_module(x)
            c = xp.ones((1,), dtype=np.float32)
            with chainer.no_backprop_mode():
                y, t, _ = eval_func((x, t, c))
                y = y.array
                grad = self.global_grad[t]
                margins = self.get_margin(
                    y, y[list(range(t.size)), t].reshape(t.size, 1), grad)
                margins = xp.min(margins, axis=1)
                margin_list.extend(list(margins.get()))

        return margin_list

    def calculate_adversarial_perturbation(self):
        print('\radversarial perturbation start', flush=True)
        iterator = self.iterator
        device = self.device or chainer.cuda.cupy.cuda.get_device_id()

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        self.attack.l2_history = list()
        for batch in it:
            x, t = self.converter(batch, device)
            self.attack(x, t)
        adv_list = self.attack.l2_history
        return adv_list
