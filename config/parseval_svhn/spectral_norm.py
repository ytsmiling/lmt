from config.parseval_svhn.base import *
from src.model.classifier import LMTraining
from src.hook.spectral_norm_regularization import SpectralNormRegularization

mode = ['default']
model = LMTraining(predictor, preprocess)
hook += [SpectralNormRegularization(0.01)]
