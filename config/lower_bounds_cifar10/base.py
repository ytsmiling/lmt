import chainer
from src.dataset.cifar10 import cifar10
from src.model.lenet import LeNet
from src.preprocess.preprocess_cifar10 import PreprocessCIFAR10
from src.extension.learning_rate_scheduler import LearningRateScheduler
from src.extension.learning_rate_scheduler import ExponentialSchedule
from src.extension.margin import Margin

batchsize = 128
dataset = cifar10()
epoch = 20
preprocess = PreprocessCIFAR10()
predictor = LeNet()
lr = 1e-2
optimizer = chainer.optimizers.MomentumSGD(lr)
extension = [(LearningRateScheduler(ExponentialSchedule(.5, (5, 10, 15))), (1, 'iteration')),
             (Margin(dataset[1], predictor, preprocess, batchsize), (epoch, 'epoch'))]
