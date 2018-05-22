import chainer
from src.dataset.mnist import mnist
from src.model.outer_polytope import OuterPolytope
from src.preprocess.preprocess_mnist import PreprocessMNIST
from src.hook.power_iteration import PowerIteration

batchsize = 50
dataset = mnist()
epoch = 20
preprocess = PreprocessMNIST()
predictor = OuterPolytope()
optimizer = chainer.optimizers.Adam(alpha=1e-3)
extension = []
hook = [PowerIteration()]
