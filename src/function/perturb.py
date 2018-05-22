import math


def perturb(arr, size, xp):
    arr += xp.random.normal(scale=size / math.sqrt(arr.size), size=arr.shape)
    arr *= 1 / (1 + size)
