from config.parseval_svhn.base import *
from src.model.wide_resnet_cb import WideResNet
from src.model.classifier import LMTraining
from src.extension.margin import Margin
from src.extension.parseval import Parseval
from src.hook.parseval_weight_decay import ParsevalWeightDecay

mode = ['parseval']
predictor = WideResNet(k=4, n_layer=16, drop=.4)
hook = [ParsevalWeightDecay(5e-4)]
model = LMTraining(predictor, preprocess)
extension += [(Parseval(beta=0.0001), (1, 'iteration')),
              (Margin(dataset[1], predictor, preprocess, batchsize), (epoch, 'epoch'))]
