#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import pathlib
import time

import chainer
from src.extension.analyze_inequality import AnalyzeInequality


def main():
    """evaluation script
    Calculate each value in inequality (5) in our paper.
    If --nograd flag is specified, only (A) and (D) are calculated.
    inequality-(attack_name).npy file is created under a specified
    result directory.

    Multi-GPU is not supported. If you want, then resort to ChainerMN.
    """

    # commandline arguments
    parser = argparse.ArgumentParser()
    # result directory
    parser.add_argument('dir', type=str,
                        help='result directory of trained model')
    parser.add_argument('attacker_config', type=str,
                        help='''please specify an attack configuration file
                        under config/attack/.''')
    # training
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-B', type=int)
    parser.add_argument('--loader_threads', '-l', type=int, default=4)
    parser.add_argument('--out', '-o', default='./result/')
    parser.add_argument('--nograd', '-n', action='store_true')
    # util
    parser.add_argument('--wait', type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise ValueError()

    # load config
    config = importlib.import_module('.'.join(args.dir.split('/') + ['config']))
    attacker_name = pathlib.Path(args.attacker_config).name
    if args.attacker_config.endswith('.py'):
        args.attacker_config = args.attacker_config[:-3]
    attacker_config = importlib.import_module(
        '.'.join(args.attacker_config.split('/')))

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
    chainer.global_config.autotune = True

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
            test_dataset, batchsize, shuffle=False, repeat=False,
            n_processes=args.loader_threads)
    else:
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batchsize, repeat=False, shuffle=False)

    # set up optimizer
    # optimizer means SGD algorithms like momentum SGD
    optimizer = config.optimizer
    optimizer.setup(model)
    for hook in getattr(config, 'hook', []):
        # hook is called before optimizer's update
        # weight decay is one of the most common optimizer hook
        optimizer.add_hook(hook)

    attack = attacker_config.attacker(model, *attacker_config.attacker_args,
                                      **attacker_config.attacker_kwargs)
    evaluator = AnalyzeInequality(
        test_iter, model, config.preprocess,
        n_class=10,  # we use datasets with 10 classes only
        attack=attack, output_dir=args.dir, device=gpu, nograd=args.nograd,
        attack_name=attacker_name)

    # my implementation switches its behavior depending on training mode
    # for details on training modes, please read codes under src/ directory
    for mode in config.mode:
        setattr(chainer.config, mode, True)

    #
    # evaluation
    #
    with chainer.using_config('cudnn_deterministic', True):
        evaluator()


if __name__ == '__main__':
    main()
