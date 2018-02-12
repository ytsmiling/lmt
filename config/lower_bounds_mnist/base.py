import chainer
from src.dataset.mnist import mnist
from src.model.lenet import LeNet
from src.preprocess.preprocess_mnist import PreprocessMNIST
from src.extension.learning_rate_scheduler import LearningRateScheduler
from src.extension.learning_rate_scheduler import ExponentialSchedule
from src.extension.margin import Margin

batchsize = 128
dataset = mnist()
epoch = 20
preprocess = PreprocessMNIST()
predictor = LeNet()
lr = 1e-1
optimizer = chainer.optimizers.MomentumSGD(lr)
extension = [(LearningRateScheduler(ExponentialSchedule(.5, (5, 10, 15))), (1, 'iteration')),
             (Margin(dataset[1], predictor, preprocess, batchsize), (epoch, 'epoch'))]
