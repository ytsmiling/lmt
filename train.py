#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import glob
import importlib
import os
import time

import chainer
from chainer.training import extensions
from src.extension.snapshot import Snapshot


def main():
    """training script

    This loads specified configuration file from config/ directory.
    Multi-GPU is not supported. If you want, then resort to ChainerMN.
    """

    # commandline arguments
    parser = argparse.ArgumentParser()
    # configuration file
    parser.add_argument('config', type=str)
    # training
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--loader_threads', '-l', type=int, default=4)
    parser.add_argument('--out', '-o', default='./result/')
    # util
    parser.add_argument('--wait', type=int)
    args = parser.parse_args()

    if args.config.endswith('.py'):
        args.config = args.config[:-3]

    # setup output directory
    prefix = os.path.join(args.out, args.config)
    cnt = len(glob.glob(prefix + '-*'))
    while True:
        output_dir = prefix + '-' + str(cnt).rjust(2, '0')
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            cnt += 1
        else:
            break

    # load config
    config = importlib.import_module('.'.join(args.config.split('/')))

    # save config
    with open(args.config + '.py', 'r') as f:
        with open(os.path.join(output_dir, 'config.py'), 'w') as wf:
            for line in f:
                wf.write(line)

    # check whether config has required information
    for name in ('batchsize', 'dataset', 'epoch', 'mode', 'model', 'optimizer'):
        assert hasattr(config, name), \
            'Configuration file do not have attribute {}!'.format(name)

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
        # if non negative GPU id is specified: use specified GPU
        # else (e.g. -1): use CPU
        chainer.cuda.get_device_from_id(gpu).use()
        chainer.cuda.set_max_workspace_size(1 * 1024 * 1024 * 1024)
    else:
        raise ValueError('currently, execution on CPU is not supported')
    chainer.global_config.autotune = True

    # set up model
    model = config.model
    if args.gpu >= 0:
        model.to_gpu()

    # get iterator of dataset
    train_dataset, val_dataset = config.dataset
    if args.loader_threads > 1:
        train_iter = chainer.iterators.MultiprocessIterator(
            train_dataset, config.batchsize, n_processes=args.loader_threads)
        val_iter = chainer.iterators.MultiprocessIterator(
            val_dataset, config.batchsize, repeat=False, n_processes=args.loader_threads)
    else:
        train_iter = chainer.iterators.SerialIterator(
            train_dataset, config.batchsize)
        val_iter = chainer.iterators.SerialIterator(
            val_dataset, config.batchsize, repeat=False)

    # set up optimizer
    # optimizer means SGD algorithms like momentum SGD
    optimizer = config.optimizer
    optimizer.setup(model)
    for hook in getattr(config, 'hook', []):
        # hook is called before optimizer's update
        # weight decay is one of the most common optimizer hook
        optimizer.add_hook(hook)

    # updater is a Chainer's training utility
    # this does the following at every iteration:
    # 1) prepare mini-batch from data iterator
    # 2) run forward and backward computation
    # 3) call optimizer (e.g. calculation of Adam)
    # 4) update parameter
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu)

    # trainer is a manager class of training
    # this invokes updater every iteration
    # this also calls extensions added later at every specified interval
    trainer = chainer.training.Trainer(updater, (config.epoch, 'epoch'), output_dir)

    # evaluator calculates accuracy and loss with network on test mode
    # usually, validation data is used for val_iter
    # in this example, I just set test data for simplicity (not recommended)
    val_interval = (1, 'epoch')
    evaluator = extensions.Evaluator(val_iter, model, device=gpu)
    trainer.extend(evaluator, trigger=val_interval, name='val')
    trainer.extend(extensions.dump_graph('main/loss'))

    #
    # additional extensions
    # learning rate scheduling is set here
    for extension, trigger in getattr(config, 'extension', []):
        trainer.extend(extension, trigger=trigger)

    # log file will be added in a result directory
    log_report_ext = extensions.LogReport(trigger=val_interval)
    trainer.extend(log_report_ext)

    # write progress of training to standard output
    trainer.extend(extensions.PrintReport([
        'elapsed_time', 'epoch', 'main/loss', 'val/main/loss',
        'main/accuracy', 'val/main/accuracy'
    ]), trigger=val_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # keep snapshot of trained model for later use like evaluation against adversarial attacks
    trainer.extend(Snapshot(), trigger=(config.epoch, 'epoch'))

    # my implementation switches its behavior depending on chainer's config
    # for details on training modes, please read codes under src/ directory
    for mode in config.mode:
        setattr(chainer.config, mode, True)

    # this is a training loop
    trainer.run()

    # training is over
    print('Result: ', output_dir, flush=True)


if __name__ == '__main__':
    main()
