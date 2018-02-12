from config.lower_bounds_cifar10.base import *
from src.model.classifier import LMTraining

mode = ['lmt', 'lmt-fc']
model = LMTraining(predictor, preprocess, c=1.)
