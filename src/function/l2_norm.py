from chainer.cuda import get_array_module
from chainer import function_node
from chainer import as_variable


class L2Norm(function_node.FunctionNode):

    def forward(self, x):
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        xp = get_array_module(*x)
        return xp.sqrt((x[0] ** 2).sum()).reshape((1,)),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0]
        norm = self.get_retained_outputs()[0]
        return as_variable(grad_outputs[0].array * x.array / norm.array),


def l2_norm(x):
    return L2Norm().apply((x,))[0]
