#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import time
import pprint

import chainer
from chainer.training import extensions


def main():
    """evaluation script

    This loads specified configuration file from config/ directory.
    Multi-GPU is not supported. If you want, then resort to ChainerMN.
    """

    # commandline arguments
    parser = argparse.ArgumentParser()
    # result directory
    parser.add_argument('dir', type=str)
    parser.add_argument('attacker_config', type=str)
    # training
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-B', type=int)
    parser.add_argument('--loader_threads', '-l', type=int, default=4)
    parser.add_argument('--out', '-o', default='./result/')
    # util
    parser.add_argument('--wait', type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError()

    # load config
    config = importlib.import_module('.'.join(args.dir.split('/') + ['config']))
    if args.attacker_config.endswith('.py'):
        args.attacker_config = args.attacker_config[:-3]
    attacker_config = importlib.import_module('.'.join(args.attacker_config.split('/')))

    # wait until specified process finish
    # this works as a pseudo job scheduler
    # Linux only
    pid = args.wait
    if pid is not None:
        while os.path.exists('/proc/{}'.format(pid)):
            time.sleep(1)

    # set up GPU
    gpu = args.gpu
    if gpu >= 0:
        # if non negative gpu id is specified: use specified gpu
        # else (e.g. -1): use cpu
        chainer.cuda.get_device_from_id(gpu).use()
        chainer.cuda.set_max_workspace_size(1 * 1024 * 1024 * 1024)
    else:
        raise ValueError('currently, execution on CPU is not supported')

    # set up model
    model = config.model
    chainer.serializers.load_npz(os.path.join(args.dir, 'snapshot.npz'), model)
    if args.gpu >= 0:
        model.to_gpu()

    # get iterator of dataset
    _, test_dataset = config.dataset
    batchsize = config.batchsize if args.batchsize is None else args.batchsize
    if args.loader_threads > 1:
        test_iter = chainer.iterators.MultiprocessIterator(
            test_dataset, batchsize, repeat=False, n_processes=args.loader_threads)
    else:
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batchsize, repeat=False)

    # set up optimizer
    # optimizer means SGD algorithms like momentum SGD
    optimizer = config.optimizer
    optimizer.setup(model)
    for hook in getattr(config, 'hook', []):
        # hook is called before optimizer's update
        # weight decay is one of the most common optimizer hook
        optimizer.add_hook(hook)

    model = attacker_config.attacker(model, *attacker_config.attacker_args, **attacker_config.attacker_kwargs)
    evaluator = extensions.Evaluator(test_iter, {'main':model, 'target':model.model}, device=gpu)

    # my implementation switches its behavior depending on training mode
    # for details on training modes, please read codes under src/ directory
    for mode in config.mode:
        setattr(chainer.config, mode, True)

    #
    # evaluation
    #
    with chainer.using_config('train', False):
        with chainer.using_config('lmt', False):
            with chainer.using_config('cudnn_deterministic', True):
                result = evaluator()
    if hasattr(model, 'save'):
        model.save(args.dir)
    for key in result.keys():
        result[key] = float(result[key])
    pprint.pprint(result)


if __name__ == '__main__':
    main()
