def register_power_iter(param):
    param.power_iteration_vector = True


class PowerIteration:
    """Support power iteration using grads"""

    name = 'PowerIteration'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self):
        pass

    def __call__(self, rule, param):
        if getattr(param, 'power_iteration_vector', False):
            param.data = param.grad
            param.grad = None
