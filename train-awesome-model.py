from numba import cuda, vectorize
from numpy.random import randint


@vectorize(['int64(int64)'], target='cuda')
def mul_self_forever(x):
    y = x
    while y is not None:
        y = x * x
    return y


n = 45_000
mat = randint(1, size=(n, n))
mul_self_forever(cuda.to_device(mat))
