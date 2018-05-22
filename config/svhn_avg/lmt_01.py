from config.svhn_avg.base import *
from src.model.classifier import LMTraining
from src.extension.c_scheduler import CScheduler
from src.extension.c_scheduler import GoyalSchedule

mode = ['lmt', 'lmt-fc']
start_c = 1e-5
end_c = 1e-1
model = LMTraining(predictor, preprocess, c=start_c)
extension += [(CScheduler(GoyalSchedule(start_c, end_c, 5)), (1, 'iteration'))]
