from config.outer_polytope.base import *
from src.model.classifier import LMTraining

mode = ['lmt', 'lmt-fc']
model = LMTraining(predictor, preprocess, c=1.)
