from chainer import cuda


class ParsevalWeightDecay(object):
    """Only the output layer is regularized.

    """

    name = 'ParsevalWeightDecay'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        if getattr(param, 'last_fc', False):
            p, g = param.data, param.grad
            with cuda.get_device_from_array(p) as dev:
                if int(dev) == -1:
                    g += self.rate * p
                else:
                    kernel = cuda.elementwise(
                        'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                    kernel(p, self.rate, g)
