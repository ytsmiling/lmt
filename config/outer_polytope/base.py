import chainer
from src.dataset.mnist import mnist
from src.model.outer_polytope import OuterPolytope
from src.preprocess.preprocess_mnist import PreprocessMNIST
from src.extension.margin import Margin

batchsize = 50
dataset = mnist()
epoch = 20
preprocess = PreprocessMNIST()
predictor = OuterPolytope()
optimizer = chainer.optimizers.Adam(alpha=1e-3)
extension = [(Margin(dataset[1], predictor, preprocess, batchsize), (epoch, 'epoch'))]
