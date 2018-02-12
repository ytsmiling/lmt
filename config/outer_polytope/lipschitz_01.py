from config.outer_polytope.base import *
from src.model.compare import Compare

mode = ['lipschitz_regularization', 'lmt']
model = Compare(predictor, preprocess, c=1e-1)
