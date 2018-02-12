from config.parseval_svhn.base import *
from src.model.wide_resnet_cb import WideResNet
from src.model.classifier import LMTraining
from src.extension.margin import Margin
from src.extension.c_scheduler import CScheduler
from src.extension.c_scheduler import GoyalSchedule
from chainer.optimizer import WeightDecay

mode = ['lmt', 'lmt-fc']
predictor = WideResNet(k=4, n_layer=16, drop=.4)
model = LMTraining(predictor, preprocess, c=1e-3)
# strong weight decay occasionally makes training fail
# even though we did not observe it with lambda=0.0001,
# if it does, removing weight decay will also work fine (with slightly worse accuracy)
hook = [WeightDecay(1e-4)]
extension += [(CScheduler(GoyalSchedule(1e-3, 1., 5)), (1, 'iteration'))]
extension += [(Margin(dataset[1], predictor, preprocess, batchsize), (epoch, 'epoch'))]
