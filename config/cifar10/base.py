import chainer
from src.model.wide_resnet_new import WideResNet
from src.dataset.cifar import cifar
from src.preprocess.preprocess_svhn import PreprocessSVHN
from src.extension.learning_rate_scheduler import LearningRateScheduler
from src.extension.learning_rate_scheduler import ExponentialSchedule
from src.hook.power_iteration import PowerIteration

batchsize = 128
dataset = cifar()
epoch = 160
preprocess = PreprocessSVHN()
predictor = WideResNet(k=4, n_layer=16, drop=.4)
lr = 1e-2
optimizer = chainer.optimizers.NesterovAG(lr)
extension = [(LearningRateScheduler(ExponentialSchedule(.1, (80, 120))), (1, 'iteration'))]
hook = [PowerIteration()]
