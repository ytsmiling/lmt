from config.parseval_svhn.base import *
from src.model.classifier import LMTraining
from src.extension.parseval import Parseval
from src.hook.parseval_weight_decay import ParsevalWeightDecay

mode = ['parseval']
model = LMTraining(predictor, preprocess)
hook = [ParsevalWeightDecay(5e-4)]
extension += [(Parseval(beta=0.0001), (1, 'iteration'))]
