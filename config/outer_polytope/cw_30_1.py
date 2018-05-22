from config.outer_polytope.base import *
from src.model.classifier import LMTraining
from config.attack.cw_30_1_1_1 import attacker, attacker_args, attacker_kwargs

attacker = attacker(predictor, *attacker_args, **attacker_kwargs)
mode = ['default']
model = LMTraining(predictor, preprocess, attacker=attacker)
attacker.register_model(model)
